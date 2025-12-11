from desc.objectives.objective_funs import _Objective
from desc.particles import _trace_particles


class DirectParticleTracing(_Objective):
    """Confinement metric for radial transport from direct tracing.

    Traces particles in flux coordinates within the equilibrium, and
    returns a confinement metric based off of the average deviation of
    the particle trajectory from its initial flux surface. The trajectories
    are traced and a line is fitted to the radial position vs time,
    and the slope of this line is used as the metric.

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    iota_grid : Grid, optional
        Grid to evaluate rotational transform profile on.
        Defaults to ``LinearGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)``.
    particles : ParticleInitializer
        should initialize them in flux coordinates, same seed
        will be used each time.
    model : TrajectoryModel
        should be either Vacuum or SlowingDown

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    _static_attrs = _Objective._static_attrs + [
        "_trace_particles",
        "_max_steps",
        "_stepsize_controller",
        "_adjoint",
        "_event",
        "_particle_chunk_size",
    ]

    _coordinates = "rtz"
    _units = "(dimensionless)"
    _print_value_fmt = "Particle Confinement error: "

    def __init__(
        self,
        eq,
        particles,
        model,
        solver=Tsit5(),  # on CPU, Tsit5(scan_kind="bounded") is recommended
        ts=jnp.arange(0, 1e-3, 100),
        stepsize_controller=None,
        adjoint=RecursiveCheckpointAdjoint(),
        max_steps=None,
        min_step_size=1e-8,
        particle_chunk_size=None,
        target=None,
        bounds=None,
        weight=1,
        normalize=False,
        normalize_target=False,
        loss_function=None,
        deriv_mode="auto",
        name="Particle Confinement",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._ts = jnp.asarray(ts)
        self._adjoint = adjoint
        if max_steps is None:
            max_steps = 1
            max_steps = int((ts[-1] - ts[0]) / min_step_size * max_steps)
        self._max_steps = max_steps
        self._min_step_size = min_step_size
        self._stepsize_controller = (
            stepsize_controller
            if stepsize_controller is not None
            else PIDController(
                rtol=1e-4,
                atol=1e-4,
                dtmin=min_step_size,
                pcoeff=0.3,
                icoeff=0.3,
                dcoeff=0,
            )
        )
        assert model.frame == "flux", "can only trace in flux coordinates"
        self._model = model
        self._particles = particles
        self._solver = solver
        self._particle_chunk_size = particle_chunk_size
        self._interpolator = FourierChebyshevField(
            L=eq.L_grid, M=eq.M_grid, N=eq.N_grid
        )
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        self._x0, self._model_args = self._particles.init_particles(
            model=self._model, field=eq
        )

        # one metric per particle
        self._dim_f = self._x0.shape[0]
        # self._dim_f = 1

        # tracing uses carteasian coordinates internally, the termainating event
        # must look at rho values by conversion
        def default_event(t, y, args, **kwargs):
            i = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
            return i > 1.0

        self._event = Event(default_event)

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")
        self._interpolator.build(eq)

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute particle tracing metric errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Average deviation in rho from initial surface, for each particle.
        """
        eq = self.things[0]
        self._interpolator.fit(params, {"iota": eq.iota, "current": eq.current})
        rpz, _ = _trace_particles(
            field=self._interpolator,
            y0=self._x0,
            model=self._model,
            model_args=self._model_args,
            ts=self._ts,
            params=None,
            stepsize_controller=self._stepsize_controller,
            saveat=SaveAt(ts=self._ts),
            max_steps=self._max_steps,
            min_step_size=self._min_step_size,
            solver=self._solver,
            adjoint=self._adjoint,
            event=self._event,
            options={},
            chunk_size=self._particle_chunk_size,
            throw=False,
            return_aux=False,
        )

        # rpz is shape [N_particles, N_time, 3], take just index rho
        rhos = rpz[:, -1, 0]
        return rhos


"""Functions for tracing particles in magnetic fields."""

import warnings
from abc import ABC, abstractmethod

import equinox as eqx
from diffrax import (
    AbstractTerm,
    Event,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from scipy.constants import Boltzmann, elementary_charge, proton_mass

from desc.backend import jax, jit, jnp, tree_map
from desc.batching import vmap_chunked
from desc.compute.utils import _compute as compute_fun
from desc.compute.utils import get_profiles, get_transforms
from desc.derivatives import Derivative
from desc.equilibrium import Equilibrium
from desc.grid import Grid, LinearGrid
from desc.io import IOAble
from desc.magnetic_fields import _MagneticField
from desc.utils import cross, dot, errorif, safediv, setdefault

JOULE_PER_EV = 11606 * Boltzmann
EV_PER_JOULE = 1 / JOULE_PER_EV


class AbstractTrajectoryModel(AbstractTerm, ABC):
    """Abstract base class for particle trajectory models.

    Subclasses should implement the ``vf`` method to compute the RHS of the ODE,
    as well as the properties `frame`, `vcoords`, `args`. ``vf`` method corresponds to
    the ``vf`` method in diffrax.AbstractTerm class and must have the same name and
    signature.
    """

    # as opposed to other classes in DESC which inherit from IOAble, this class
    # is a subclass of diffrax.AbstractTerm which is an Equinox.Module. The following
    # attributes need to be defined as static fields for JAX transformation.
    _frame: str = eqx.field(static=True)
    vcoords: list[str] = eqx.field(static=True)
    args: list[str] = eqx.field(static=True)

    @property
    @abstractmethod
    def frame(self):
        """One of "flux" or "lab", indicating which frame the model is defined in.

        "flux" traces particles in (rho, theta, zeta) magnetic coordinates
        "lab" traces particles in (R, phi, Z) lab frame

        """
        return self._frame

    @property
    @abstractmethod
    def vcoords(self):  # noqa : F811
        """Velocity coordinates used by the model, in order.

        Options are:
        "v" : modulus of velocity
        "vpar" : velocity in direction of local magnetic field.
        "vperp" : modulus of velocity perpendicular to local magnetic field.
        "vR" : velocity in lab frame R direction
        "vP" : velocity in lab frame phi direction
        "vZ" : velocity in lab frame Z direction
        """
        pass

    @property
    @abstractmethod
    def args(self):  # noqa : F811
        """Additional arguments needed by the model.

        Eg, "m", "q", "mu", for mass, charge, magnetic moment (mv⊥²/2|B|).
        """
        pass

    @abstractmethod
    def vf(self, t, x, args):
        """RHS of the particle trajectory ODE."""
        pass

    def contr(self, t0, t1, **kwargs):
        """Needed by diffrax."""
        return t1 - t0

    def prod(self, vf, control):
        """Needed by diffrax."""

        def _mul(v):
            return control * v

        return tree_map(_mul, vf)


class VacuumGuidingCenterTrajectory(AbstractTrajectoryModel):
    """Guiding center trajectories in vacuum, conserving energy and mu.

    Solves the following ODEs,

    d𝐑/dt = v∥ 𝐛 + (m / q B²) ⋅ (v∥² + 1/2 v⊥²) ( 𝐛 × ∇B )

    dv∥/dt = − (v⊥² / 2B) ( 𝐛 ⋅ ∇B )

    where 𝐁 is the magnetic field vector at position 𝐑, B is the magnitude of
    the magnetic field and 𝐛 is the unit magnetic field 𝐁/B.

    Parameters
    ----------
    frame : {"lab", "flux"}
        Which coordinate frame is used for tracing particles. 'lab' corresponds to
        {R, phi, Z} coordinates, 'flux' corresponds to {rho, theta, zeta} coordinates.
        Frame must be compatible with the source of the field, i.e. if tracing in an
        Equilibrium, set frame="flux" or if tracing in a MagneticField, choose "lab".

        Although particles can be traced in "lab" frame using an Equilibrium, it is
        not recommended, since it requires coodinate mapping at each step of the
        integration. Thus, it is not implemented. For that case, we recommend converting
        the final output to "lab" frame after the integration is done using
        Equilibrium.map_coordinates method.

        If anytime during the integration, the particle's rho coordinate becomes
        smaller than 1e-6, the magnetic field is computed at rho = 1e-6 to avoid
        numerical issues with the rho = 0. Note that particle position doesn't have
        discontinuity due to this, only the magnetic field is computed at a different
        point.
    """

    vcoords = ["vpar"]
    args = ["m", "q", "mu"]

    def __init__(self, frame):
        assert frame in ["lab", "flux"]
        self._frame = frame

    @property
    def frame(self):
        """Coordinate frame of the model."""
        return self._frame

    @jit
    def vf(self, t, x, args):
        """RHS of guiding center trajectories without collisions or slowing down.

        ``vf`` method corresponds to the ``vf`` method in diffrax.AbstractTerm class
        and must have the same name and signature.

        Parameters
        ----------
        t : float
            Time to evaluate RHS at.
        x : jax.Array, shape(4,)
            Position of particle in phase space [rho, theta, zeta, vpar] or
            [R, phi, Z, vpar].
        args : tuple
            Should include the arguments needed by the model, (m, q, mu) as
            an array, Equilibrium or MagneticField object, params and any additional
            keyword arguments needed for magnetic field computation, such as iota
            profile for the Equilibrium, and source_grid for the MagneticField.

        Returns
        -------
        dx : jax.Array, shape(N,4)
            Velocity of particles in phase space.
        """
        x = x.squeeze()
        model_args, eq_or_field, params, kwargs = args
        m, q, mu = model_args
        if self.frame == "flux":
            assert isinstance(eq_or_field, (Equilibrium, FourierChebyshevField)), (
                "Integration in flux coordinates requires an Equilibrium or "
                "FourierChebyshevField."
            )
            if isinstance(eq_or_field, FourierChebyshevField):
                return self._compute_flux_coordinates_with_fit(x, eq_or_field, m, q, mu)

    def _compute_flux_coordinates_with_fit(self, x, field, m, q, mu):
        """ODE equation for vacuum guiding center in flux coordinates.

        A Fourier-Chebyshev fit for each component of the 3D magnetic field B, gradient
        of the magnetic field strength and the basis vectors must be given as real and
        imaginary parts. Basis vector e^theta is not well around the axis and blows up,
        instead we use e^theta*rho which results in better fit.
        """
        xp, yp, zeta, vpar = x
        rho = jnp.sqrt(xp**2 + yp**2)
        theta = jnp.arctan2(yp, xp)
        # compute functions are not correct for very small rho
        rho = jnp.where(rho < 1e-6, 1e-6, rho)

        data = field.evaluate(rho, theta, zeta)

        Rdot = vpar * data["b"] + (
            (m / q / data["|B|"] ** 2)
            * ((mu * data["|B|"] / m) + vpar**2)
            * cross(data["b"], data["grad(|B|)"])
        )
        # take dot product for rho, theta and zeta coordinates
        rhodot = dot(Rdot, data["e^rho"])
        thetadot_x_rho = dot(Rdot, data["e^theta*rho"])
        zetadot = dot(Rdot, data["e^zeta"])

        # get the derivative for cartesian-like coordinates
        xpdot = rhodot * jnp.cos(theta) - thetadot_x_rho * jnp.sin(theta)
        ypdot = rhodot * jnp.sin(theta) + thetadot_x_rho * jnp.cos(theta)
        # derivative the parallel velocity
        vpardot = -mu / m * dot(data["b"], data["grad(|B|)"])
        dxdt = jnp.array([xpdot, ypdot, zetadot, vpardot]).reshape(x.shape)
        return dxdt.squeeze()


class AbstractParticleInitializer(IOAble, ABC):
    """Abstract base class for initial distribution of particles for tracing.

    Subclasses should implement the `init_particles` method.
    """

    @abstractmethod
    def init_particles(self, model, field):
        """Initialize a distribution of particles.

        Should return two things:
        - an NxD array of initial particle positions and velocities,
        where N is the number of particles and D is the dimensionality of the
        trajectory model (4, 5, or 6).
        - a tuple of additional arguments requested by the model, eg mass, charge,
        magnetic moment of each particle.
        """
        pass

    def _return_particles(self, x, v, vpar, model, field, params=None, **kwargs):
        """Return the particles in a common format.

        Parameters
        ----------
        x : jax.Array, shape(N,3)
            Initial particle positions in either flux (rho, theta, zeta) coordinates or
            cylindirical (lab) coordinates, shape (N, 3), where N is the number of
            particles.
        v : ArrayLike, shape(N,)
            Initial particle speeds
        vpar : ArrayLike, shape(N,)
            Initial particle parallel velocities, in the direction of local magnetic
            field.
        model : AbstractTrajectoryModel
            Model to use for tracing particles, which defines the frame and
            velocity coordinates.
        field : Equilibrium or _MagneticField
            Source of magnetic field to use for tracing particles.

        Returns
        -------
        x0 : jax.Array, shape(N,D)
            Initial particle positions and velocities, where D is the dimensionality of
            the trajectory model, which includes 3D spatial dimensions and depending on
            the model, parallel velocity and total velocity. The initial positions are
            in the frame of the model.
        args : jax.Array, shape(N,M)
            Additional arguments needed by the model, such as mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle. M is the number of arguments
            requested by the model which is equal to len(model.args). N is the number
            of particles.
        """
        vs = []
        for vcoord in model.vcoords:
            if vcoord == "vpar":
                vs.append(vpar)
            elif vcoord == "v":
                vs.append(v)
            else:
                raise NotImplementedError
        v0 = jnp.array(vs).T

        args = []
        for arg in model.args:
            if arg == "m":
                args += [self.m]
            elif arg == "q":
                args += [self.q]
            elif arg == "mu":
                vperp2 = v**2 - vpar**2
                modB = _compute_modB(x, field, params, **kwargs)
                args += [self.m * vperp2 / (2 * modB)]

        args = jnp.array(args).T
        return jnp.hstack([x, v0]), args


def _compute_modB(x, field, params, **kwargs):
    if isinstance(field, Equilibrium):
        grid = Grid(
            x,
            spacing=jnp.zeros_like(x),
            sort=False,
            NFP=field.NFP,
            jitable=True,
        )
        profiles = get_profiles("|B|", field, grid)
        transforms = get_transforms("|B|", field, grid, jitable=True)
        if "iota" in kwargs:
            profiles["iota"] = kwargs["iota"]
        return compute_fun(
            field,
            "|B|",
            params=params,
            grid=grid,
            profiles=profiles,
            transforms=transforms,
        )["|B|"]
    source_grid = kwargs.pop("source_grid", None)
    return jnp.linalg.norm(
        field.compute_magnetic_field(x, params=params, source_grid=source_grid), axis=-1
    )


class ManualParticleInitializerFlux(AbstractParticleInitializer):
    """Manually specify particle starting positions and energy in flux coordinates.

    Parameters
    ----------
    rho0 : array-like
        Initial radial coordinates
    theta0 : array-like
        Initial poloidal coordinates in radians
    zeta0 : array-like
        Initial toroidal coordinates in radians
    xi0 : array-like
        Initial normalized parallel velocity, xi=vpar/v
    E : array-like
        Initial particle kinetic energy, in eV
    m : array-like
        Particle mass, in proton masses
    q : array-like
        Particle charge, in units of elementary charge.
    """

    def __init__(
        self,
        rho0,
        theta0,
        zeta0,
        xi0,
        E=3.5e6,
        m=4,
        q=2,
    ):
        rho0, theta0, zeta0, xi0, E, m, q = map(
            jnp.atleast_1d, (rho0, theta0, zeta0, xi0, E, m, q)
        )
        rho0, theta0, zeta0, xi0, E, m, q = jnp.broadcast_arrays(
            rho0, theta0, zeta0, xi0, E, m, q
        )
        self.m = m * proton_mass
        self.q = q * elementary_charge
        self.rho0 = rho0
        self.theta0 = theta0
        self.zeta0 = zeta0
        self.v0 = jnp.sqrt(2 * E * JOULE_PER_EV / self.m)
        self.vpar0 = xi0 * self.v0

        errorif(
            any(jnp.logical_or(self.rho0 > 1.0, self.rho0 < 0.0)),
            ValueError,
            "Flux coordinate rho must be between 0 and 1.",
        )

    def init_particles(self, model, field, **kwargs):
        """Initialize particles for a given trajectory model.

        Parameters
        ----------
        model : AbstractTrajectoryModel
            Model to use for tracing particles, which defines the frame and
            velocity coordinates.
        field : Equilibrium or _MagneticField
            Source of magnetic field to use for tracing particles.
        kwargs : dict, optional
            source_grid for the magnetic field computation, if using a MagneticField
            object, can be passed as a keyword argument.
            If you are trying to initialize particles in lab coordinates for a
            MagneticField from inputs in flux coordinates, you must pass the
            Equilibrium as keyword argument "eq" in kwargs.

        Returns
        -------
        x0 : jax.Array, shape(N,D)
            Initial particle positions and velocities, where D is the dimensionality of
            the trajectory model, which includes 3D spatial dimensions and depending on
            the model, parallel velocity and total velocity. The initial positions are
            in the frame of the model.
        args : jax.Array, shape(N,M)
            Additional arguments needed by the model, such as mass, charge, and
            magnetic moment (mv⊥²/2|B|) of each particle. M is the number of arguments
            requested by the model which is equal to len(model.args). N is the number
            of particles.
        """
        x = jnp.array([self.rho0, self.theta0, self.zeta0]).T
        params = field.params_dict
        if model.frame == "flux":
            if not isinstance(field, Equilibrium):
                raise ValueError(
                    "Please use Equilibrium object with the model in flux frame."
                )
            x = x
            if field.iota is None:
                iota = field.get_profile("iota")
                params["i_l"] = iota.params
                kwargs["iota"] = iota
        elif model.frame == "lab":
            if isinstance(field, Equilibrium):
                raise NotImplementedError(
                    "If you have an Equilibrium object, you should use the model "
                    "in flux frame. Since trying to integrate in lab frame will "
                    "require multiple coordinate mapping, it is not implemented."
                )
            elif isinstance(field, _MagneticField):
                eq = kwargs.pop("eq", None)
                if isinstance(eq, Equilibrium):
                    grid = Grid(
                        x,
                        spacing=jnp.zeros_like(x),
                        sort=False,
                        NFP=eq.NFP,
                        jitable=True,
                    )
                    x = eq.compute("x", grid=grid)["x"]
                else:
                    raise NotImplementedError(
                        "The given model requires inputs in lab coordinates. Without "
                        "an equilibrium, converting the initial positions from flux "
                        "to lab frame is not possible. You can pass the Equilibrium "
                        "as a keyword argument 'eq' in kwargs and it will be used for "
                        "the mapping."
                    )
        else:
            raise NotImplementedError

        return super()._return_particles(
            x=x,
            v=self.v0,
            vpar=self.vpar0,
            model=model,
            field=field,
            params=params,
            **kwargs,
        )


def _trace_particles(
    field,
    y0,
    model,
    model_args,
    ts,
    params,
    max_steps,
    min_step_size,
    stepsize_controller,
    saveat,
    solver,
    adjoint,
    event,
    options,
    chunk_size=None,
    throw=False,
    return_aux=False,
):
    """Trace charged particles in an equilibrium or external magnetic field.

    This is the jit friendly version of the `trace_particles` function. For full
    documentation, see `trace_particles`. This function takes the outputs of
    `initializer.init_particles` as inputs, rather than the particle initializer
    itself. There won't be any checks on the y0 and model_args inputs, so make sure
    they are in the correct format. One can use this function in an objective. All
    the arguments must be passed with a value, see ``trace_particles`` for default
    values for common use.

    Parameters
    ----------
    y0 : array-like
        Initial particle positions and velocities, stacked in horizontally [x0, v0].
        The first output of `initializer.init_particles`.
    model_args : array-like
        Additional arguments needed by the model, such as mass, charge, and
        magnetic moment (mv⊥²/2|B|) of each particle. The second output of
        `initializer.init_particles`.
    stepsize_controller : diffrax.AbstractStepsizeController
        Stepsize controller to use for the integration.
    saveat : diffrax.SaveAt
        SaveAt object to specify where to save the output.
    event : diffrax.Event
        Custom event function to stop integration.
    """
    # convert cartesian-like for integration in flux coordinates
    if isinstance(field, (Equilibrium, FourierChebyshevField)):
        xp = y0[:, 0] * jnp.cos(y0[:, 1])
        yp = y0[:, 0] * jnp.sin(y0[:, 1])
        y0 = y0.at[:, 0].set(xp)
        y0 = y0.at[:, 1].set(yp)

    # suppress warnings till its fixed upstream:
    # https://github.com/patrick-kidger/diffrax/issues/445
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="unhashable type")
        # we only want to map over initial positions and particle arguments
        # Note: vmap with keyword arguments is weird, not using it for now
        out = vmap_chunked(
            _intfun_wrapper, in_axes=(0, 0) + 13 * (None,), chunk_size=chunk_size
        )(
            y0,
            model_args,
            field,
            params,
            ts,
            max_steps,
            min_step_size,
            solver,
            stepsize_controller,
            adjoint,
            event,
            model,
            saveat,
            options,
            throw,
        )
    yt = out.ys
    yt = jnp.where(jnp.isinf(yt), jnp.nan, yt)

    x = yt[:, :, :3]
    v = yt[:, :, 3:]

    # convert back to flux coordinates
    if isinstance(field, (Equilibrium, FourierChebyshevField)):
        rho = jnp.sqrt(x[:, :, 0] ** 2 + x[:, :, 1] ** 2)
        theta = jnp.arctan2(x[:, :, 1], x[:, :, 0])
        theta = jnp.where(theta < 0, theta + 2 * jnp.pi, theta)
        x = x.at[:, :, 0].set(rho)
        x = x.at[:, :, 1].set(theta)

    if not return_aux:
        return x, v
    else:
        return x, v, (out.ts, out.stats, out.result)


def _intfun_wrapper(
    x,
    model_args,
    field,
    params,
    ts,
    max_steps,
    min_step_size,
    solver,
    stepsize_controller,
    adjoint,
    event,
    model,
    saveat,
    options,
    throw,
):
    """Wrapper for the integration function for vectorized inputs.

    Defining a lambda function inside the `_trace_particles` function leads
    to recompilations, so instead we define the wrapper here.
    """
    return diffeqsolve(
        terms=model,
        solver=solver,
        y0=x,
        args=[model_args, field, params, options],
        t0=ts[0],
        t1=ts[-1],
        saveat=saveat,
        max_steps=max_steps,
        dt0=min_step_size,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        event=event,
        throw=throw,
    )


class FourierChebyshevField(IOAble):
    """Convenience class for fitting and evaluating equilibrium fields.

    This class is intended to be used during particle tracing to reduce overhead
    of creating transforms. It fits a Fourier-Fourier-Chebyshev series to the
    quantities required for guiding center equations, and evaluates them
    at requested points during tracing.

    Parameters
    ----------
    L : int
        Maximum order of the Chebyshev polynomial to be used in the radial direction.
    M : int
        Maximum order of the Fourier series to be used in the poloidal direction.
    N : int
        Maximum order of the Fourier series to be used in the toroidal direction.
    """

    _static_attrs = ["L", "M", "N", "M_fft", "N_fft", "data_keys"]

    def __init__(self, L, M, N):
        self.L = L
        self.M = M
        self.N = N

    def build(self, eq):
        """Build the constants for fit.

        During optimization, equilibrium field changes, however, the same transforms
        can be used to get the fit faster. This method creates the grid and transforms
        to be used during the fitting procedure.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium to be used to get transforms.

        """
        self.data_keys = ["B", "grad(|B|)", "e^rho", "e^theta*rho", "e^zeta"]
        self.l = jnp.arange(self.L)
        self.M_fft = 2 * self.M + 1
        self.N_fft = 2 * self.N + 1
        self.m = jnp.fft.fftfreq(self.M_fft) * self.M_fft
        self.n = jnp.fft.fftfreq(self.N_fft) * self.N_fft
        x = jnp.cos(jnp.pi * (2 * self.l + 1) / (2 * self.L))
        rho = (x + 1) / 2
        self.grid = LinearGrid(rho=rho, M=self.M, N=self.N, sym=False, NFP=eq.NFP)
        self.transforms = get_transforms(self.data_keys, eq, self.grid)

    def fit(self, params, profiles):
        """Fit a Fourier-Chebyshev series to an equilibrium field.

        First computes the magnetic field, its gradient and basis vectors at
        the grid created in build. Then, finds the spectral coefficients to
        each component of the computed vectors. Since e^theta doesn't behave
        well around axis, the fit is computed for e^theta*rho (which is what actually
        required by the guiding center equations).

        Parameters
        ----------
        params : dict
            Equilibriums `params_dict` which contains the parameters that define
            the equiliubrium.
        profiles : dict of Profiles
            Profiles necessary to compute magnetic field. Either iota or current
            profile must be given.

        """
        data_raw = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self.data_keys,
            params,
            self.transforms,
            profiles,
        )
        L, M, N = self.L, self.M_fft, self.N_fft
        # TODO: e^zeta only has _p component, rest is 0, no need to fit
        keys = [key + i for key in self.data_keys for i in ["_r", "_p", "_z"]]
        # stack data to perform 15 transforms in batch
        stacked_data = jnp.stack(
            [
                data_raw[key][:, i].reshape(N, L, M)
                for key in self.data_keys
                for i in [0, 1, 2]
            ]
        )
        coefs = jax.scipy.fft.dct(stacked_data, axis=2, norm=None)
        # handle the 0-th Chebyshev coefficient and normalization
        coefs = coefs.at[:, :, 0, :].divide(2)
        coefs /= self.L

        coefs = jnp.fft.fft(coefs, axis=3, norm=None)
        coefs = jnp.fft.fft(coefs, axis=1, norm=None)

        data = {}
        coefs_real = coefs.real
        coefs_imag = coefs.imag

        for i, key in enumerate(keys):
            data[key + "_real"] = coefs_real[i]
            data[key + "_imag"] = coefs_imag[i]

        data["l"] = self.l
        data["m"] = self.m
        data["n"] = self.n
        data["M"] = self.M_fft
        data["N"] = self.N_fft

        self.params_dict = data

    def evaluate(self, rho, theta, zeta, params=None):
        """Evaluate the Fourier-Chebyshev series at a point.

        Parameters
        ----------
        rho, theta, zeta : float
            Radial, poloidal and toroidal coordinates to evaluate.
        params : dict
            The spectral coefficients obtained from `FourierChebyshevField.fit()`
            which is stored as `self.params`.
        """
        if params is None:
            params = self.params_dict

        # the cosine transforms reverses the order
        r0p = 1 - 2 * rho
        Tl = jnp.cos(params["l"] * jnp.arccos(r0p))
        m_theta = params["m"] * theta
        expm_real = jnp.cos(m_theta) / params["M"]
        expm_imag = jnp.sin(m_theta) / params["M"]

        zeta = (zeta * self.grid.NFP) % (2 * jnp.pi)
        n_zeta = params["n"] * zeta
        expn_real = jnp.cos(n_zeta) / params["N"]
        expn_imag = jnp.sin(n_zeta) / params["N"]

        # The new shape for these arrays will be (k, n, l, m) where k=15
        cf_real_all = jnp.stack(
            [
                params[key + i + "_real"]
                for key in self.data_keys
                for i in ["_r", "_p", "_z"]
            ]
        )
        cf_imag_all = jnp.stack(
            [
                params[key + i + "_imag"]
                for key in self.data_keys
                for i in ["_r", "_p", "_z"]
            ]
        )

        # "knlm,l->knm" contracts the 'l' dimension for all 'k' batches at once
        f_l_real = jnp.einsum("knlm,l->knm", cf_real_all, Tl)
        f_l_imag = jnp.einsum("knlm,l->knm", cf_imag_all, Tl)

        # "knm,m->kn" contracts the 'm' dimension for all 'k' batches
        f_lm_real = jnp.einsum("knm,m->kn", f_l_real, expm_real) - jnp.einsum(
            "knm,m->kn", f_l_imag, expm_imag
        )
        f_lm_imag = jnp.einsum("knm,m->kn", f_l_real, expm_imag) + jnp.einsum(
            "knm,m->kn", f_l_imag, expm_real
        )

        # "kn,n->k" contracts the 'n' dimension, leaving just the batch dimension
        results = jnp.einsum("kn,n->k", f_lm_real, expn_real) - jnp.einsum(
            "kn,n->k", f_lm_imag, expn_imag
        )

        out = {}
        # Magnetic Field B
        B = results[0:3]
        out["|B|"] = jnp.linalg.norm(B)
        out["b"] = B / out["|B|"]
        # grad(|B|)
        out["grad(|B|)"] = results[3:6]
        # e^rho
        out["e^rho"] = results[6:9]
        # e^theta*rho
        out["e^theta*rho"] = results[9:12]
        # e^zeta
        out["e^zeta"] = results[12:15]

        return out


"""Base classes for objectives."""

import functools
from abc import ABC, abstractmethod

import numpy as np

from desc.backend import (
    desc_config,
    execute_on_cpu,
    jit,
    jnp,
    tree_flatten,
    tree_map,
    tree_unflatten,
    use_jax,
)
from desc.batching import batched_vectorize
from desc.derivatives import Derivative
from desc.io import IOAble
from desc.optimizable import Optimizable
from desc.utils import (
    PRINT_WIDTH,
    Timer,
    ensure_tuple,
    errorif,
    flatten_list,
    is_broadcastable,
    isposint,
    setdefault,
    unique_list,
    warnif,
)

doc_target = """
    target : {float, ndarray}, optional
        Target value(s) of the objective. Only used if ``bounds`` is ``None``.
        Must be broadcastable to ``Objective.dim_f``.
"""
doc_bounds = """
    bounds : tuple of {float, ndarray}, optional
        Lower and upper bounds on the objective. Overrides ``target``.
        Both bounds must be broadcastable to ``Objective.dim_f``.
"""
doc_weight = """
    weight : {float, ndarray}, optional
        Weighting to apply to the Objective, relative to other Objectives.
        Must be broadcastable to ``Objective.dim_f``.
"""
doc_normalize = """
    normalize : bool, optional
        Whether to compute the error in physical units or non-dimensionalize.
"""
doc_normalize_target = """
    normalize_target : bool, optional
        Whether target and bounds should be normalized before comparing to computed
        values. If ``normalize`` is ``True`` and the target is in physical units,
        this should also be set to ``True``.
"""
doc_loss_function = """
    loss_function : {None, 'mean', 'min', 'max','sum'}, optional
        Loss function to apply to the objective values once computed. This loss function
        is called on the raw compute value, before any shifting, scaling, or
        normalization.
"""
doc_deriv_mode = """
    deriv_mode : {"auto", "fwd", "rev"}
        Specify how to compute Jacobian matrix, either forward mode or reverse mode AD.
        ``auto`` selects forward or reverse mode based on the size of the input and
        output of the objective. Has no effect on ``self.grad`` or ``self.hess`` which
        always use reverse mode and forward over reverse mode respectively.
"""
doc_name = """
    name : str, optional
        Name of the objective.
"""
doc_jac_chunk_size = """
    jac_chunk_size : int or ``auto``, optional
        Will calculate the Jacobian
        ``jac_chunk_size`` columns at a time, instead of all at once.
        The memory usage of the Jacobian calculation is roughly
        ``memory usage = m0+m1*jac_chunk_size``: the smaller the chunk size,
        the less memory the Jacobian calculation will require (with some baseline
        memory usage). The time it takes to compute the Jacobian is roughly
        ``t = t0+t1/jac_chunk_size`` so the larger the ``jac_chunk_size``, the faster
        the calculation takes, at the cost of requiring more memory.
        If ``None``, it will use the largest size i.e ``obj.dim_x``.
        Can also help with Hessian computation memory, as Hessian is essentially
        ``jacfwd(jacrev(f))``, and each of these operations may be chunked.
        Defaults to ``chunk_size=None``.
        Note: When running on a CPU (not a GPU) on a HPC cluster, DESC is unable to
        accurately estimate the available device memory, so the ``auto`` chunk_size
        option will yield a larger chunk size than may be needed. It is recommended
        to manually choose a chunk_size if an OOM error is experienced in this case.
"""
docs = {
    "target": doc_target,
    "bounds": doc_bounds,
    "weight": doc_weight,
    "normalize": doc_normalize,
    "normalize_target": doc_normalize_target,
    "loss_function": doc_loss_function,
    "deriv_mode": doc_deriv_mode,
    "name": doc_name,
    "jac_chunk_size": doc_jac_chunk_size,
}


# Note: If we ever switch to Python 3.13 for building the docs, there will probably
# be some errors since 3.13 changed how tabs are handled in docstrings. This can be
# resolved by deleting the tabs in the collected docstring above and the ones
# that are defined in objectives. Check `test_objective_docstring`.
def collect_docs(
    overwrite=None,
    target_default="",
    bounds_default="",
    normalize_detail=None,
    normalize_target_detail=None,
    loss_detail=None,
    coil=False,
):
    """Collect default parameters for the docstring of Objective.

    Parameters
    ----------
    overwrite : dict, optional
        Dict of strings to overwrite from the ``_Objective``'s docstring. If None,
        all default parameters are included as they are. Use this argument if
        you want to specify a special docstring for a specific parameter in
        your objective definition.
    target_default : str, optional
        Default value for the ``target`` parameter.
    bounds_default : str, optional
        Default value for the ``bounds`` parameter.
    normalize_detail : str, optional
        Additional information about the ``normalize`` parameter.
    normalize_target_detail : str, optional
        Additional information about the ``normalize_target`` parameter.
    loss_detail : str, optional
        Additional information about the ``loss`` function.
    coil : bool, optional
        Whether the objective is a coil objective. If ``True``, adds extra docs
        to ``target`` and ``loss_function``.

    Returns
    -------
    doc_params : str
        String of default parameters for the docstring.

    """
    doc_params = ""
    for key in docs.keys():
        if overwrite is not None and key in overwrite.keys():
            doc_params += overwrite[key].rstrip()
        else:
            if key == "target":
                target = ""
                if coil:
                    target += (
                        "If array, it has to be flattened according to the "
                        + "number of inputs."
                    )
                if target_default != "":
                    target = target + " Defaults to " + target_default
                doc_params += docs[key].rstrip() + target
            elif key == "bounds" and bounds_default != "":
                doc_params = (
                    doc_params + docs[key].rstrip() + " Defaults to " + bounds_default
                )
            elif key == "loss_function":
                loss = ""
                if coil:
                    loss = " Operates over all coils, not each individual coil."
                if loss_detail is not None:
                    loss += loss_detail
                doc_params += docs[key].rstrip() + loss
            elif key == "normalize":
                norm = ""
                if normalize_detail is not None:
                    norm += normalize_detail
                doc_params += docs[key].rstrip() + norm
            elif key == "normalize_target":
                norm_target = ""
                if normalize_target_detail is not None:
                    norm_target = normalize_target_detail
                doc_params += docs[key].rstrip() + norm_target
            else:
                doc_params += docs[key].rstrip()

    return doc_params


class ObjectiveFunction(IOAble):
    """Objective function comprised of one or more Objectives.

    Parameters
    ----------
    objectives : tuple of Objective
        List of objectives to be minimized.
    use_jit : bool, optional
        Whether to just-in-time compile the objectives and derivatives.
    deriv_mode : {"auto", "batched", "blocked"}
        Method for computing Jacobian matrices. ``batched`` uses forward mode, applied
        to the entire objective at once, and is generally the fastest for vector
        valued objectives. Its memory intensity vs. speed may be traded off through
        the ``jac_chunk_size`` keyword argument. "blocked" builds the Jacobian for
        each objective separately, using each objective's preferred AD mode (and
        each objective's `jac_chunk_size`). Generally the most efficient option when
        mixing scalar and vector valued objectives.
        ``auto`` defaults to ``batched`` if all sub-objectives are set to ``fwd``,
        otherwise ``blocked``.
    name : str
        Name of the objective function.
    jac_chunk_size : int or ``auto``, optional
         If ``batched`` deriv_mode is used, will calculate the Jacobian
        ``jac_chunk_size`` columns at a time, instead of all at once.
        The memory usage of the Jacobian calculation is roughly
        ``memory usage = m0+m1*jac_chunk_size``: the smaller the chunk size,
        the less memory the Jacobian calculation will require (with some baseline
        memory usage). The time it takes to compute the Jacobian is roughly
        ``t = t0+t1/jac_chunk_size`` so the larger the ``jac_chunk_size``, the faster
        the calculation takes, at the cost of requiring more memory.
        If ``None``, it will use the largest size i.e ``obj.dim_x``.
        Can also help with Hessian computation memory, as Hessian is essentially
        ``jacfwd(jacrev(f))``, and each of these operations may be chunked.
        Defaults to ``chunk_size="auto"``.
        Note: When running on a CPU (not a GPU) on a HPC cluster, DESC is unable to
        accurately estimate the available device memory, so the "auto" chunk_size
        option will yield a larger chunk size than may be needed. It is recommended
        to manually choose a chunk_size if an OOM error is experienced in this case.

    """

    _io_attrs_ = [
        "_deriv_mode",
        "_jac_chunk_size",
        "_name",
        "_objectives",
        "_use_jit",
    ]
    _static_attrs = [
        "_built",
        "_compile_mode",
        "_compiled",
        "_deriv_mode",
        "_jac_chunk_size",
        "_name",
        "_things_per_objective_idx",
        "_use_jit",
        "_static_attrs",
    ]

    def __init__(
        self,
        objectives,
        use_jit=True,
        deriv_mode="auto",
        name="ObjectiveFunction",
        jac_chunk_size="auto",
    ):
        if not isinstance(objectives, (tuple, list)):
            objectives = (objectives,)
        assert all(
            isinstance(obj, _Objective) for obj in objectives
        ), "members of ObjectiveFunction should be instances of _Objective"
        assert use_jit in {True, False}
        if deriv_mode == "looped":
            # overwrite the user inputs if deprecated "looped" was given
            warnif(
                True,
                DeprecationWarning,
                '``deriv_mode="looped"`` is deprecated in favor of'
                ' ``deriv_mode="batched"`` with ``jac_chunk_size=1``.',
            )
            deriv_mode = "batched"
            jac_chunk_size = 1
        assert deriv_mode in {"auto", "batched", "blocked"}
        assert jac_chunk_size in ["auto", None] or isposint(jac_chunk_size)

        self._jac_chunk_size = jac_chunk_size
        self._objectives = objectives
        self._use_jit = use_jit
        self._deriv_mode = deriv_mode
        self._built = False
        self._compiled = False
        self._name = name

    def _unjit(self):
        """Remove jit compiled methods."""
        methods = [
            "compute_scaled",
            "compute_scaled_error",
            "compute_unscaled",
            "compute_scalar",
            "jac_scaled",
            "jac_scaled_error",
            "jac_unscaled",
            "hess",
            "grad",
            "jvp_scaled",
            "jvp_scaled_error",
            "jvp_unscaled",
            "vjp_scaled",
            "vjp_scaled_error",
            "vjp_unscaled",
        ]
        for method in methods:
            try:
                setattr(
                    self, method, functools.partial(getattr(self, method)._fun, self)
                )
                if method not in self._static_attrs:
                    self._static_attrs += [method]
            except AttributeError:
                pass

    @execute_on_cpu
    def build(self, use_jit=None, verbose=1):
        """Build the objective.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        if use_jit is not None:
            self._use_jit = use_jit
        timer = Timer()
        timer.start("Objective build")

        # build objectives
        self._dim_f = 0
        for objective in self.objectives:
            if not objective.built:
                if verbose > 0:
                    print("Building objective: " + objective.name)
                objective.build(use_jit=self.use_jit, verbose=verbose)
            self._dim_f += objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        self._set_things()

        # setting derivative mode and chunking.
        sub_obj_jac_chunk_sizes_are_ints = [
            isposint(obj._jac_chunk_size) for obj in self.objectives
        ]
        sub_obj_chunk_sizes_names = [
            (obj.__class__.__name__, obj._jac_chunk_size) for obj in self.objectives
        ]
        errorif(
            any(sub_obj_jac_chunk_sizes_are_ints) and self._deriv_mode == "batched",
            ValueError,
            "'jac_chunk_size' was passed into one or more sub-objectives, but the\n"
            "ObjectiveFunction is using 'batched' deriv_mode, so sub-objective \n"
            "'jac_chunk_size' will be ignored in favor of the ObjectiveFunction's \n"
            f"'jac_chunk_size' of {self._jac_chunk_size}.\n"
            "Specify 'blocked' deriv_mode and don't pass `jac_chunk_size` for \n"
            "ObjectiveFunction if each sub-objective is desired to have a \n"
            "different 'jac_chunk_size' for its Jacobian computation. \n"
            "`jac_chunk_size` of sub-objective(s): \n"
            f"{sub_obj_chunk_sizes_names}\n"
            f"Note: If you didn't specify 'jac_chunk_size' for the sub-objectives, \n"
            "it might be that sub-objective has an internal logic to determine the \n"
            "chunk size based on the available memory.",
        )

        if self._deriv_mode == "auto":
            if all((obj._deriv_mode == "fwd") for obj in self.objectives) and not any(
                sub_obj_jac_chunk_sizes_are_ints
            ):
                self._deriv_mode = "batched"
            else:
                self._deriv_mode = "blocked"

        errorif(
            isposint(self._jac_chunk_size) and self._deriv_mode in ["blocked"],
            ValueError,
            "'jac_chunk_size' was passed into ObjectiveFunction, but the "
            "ObjectiveFunction is not using 'batched' deriv_mode",
        )

        if self._jac_chunk_size == "auto":
            # Heuristic estimates of fwd mode Jacobian memory usage,
            # slightly conservative, based on using ForceBalance as the objective
            estimated_memory_usage = 2.4e-7 * self.dim_f * self.dim_x + 1  # in GB
            max_chunk_size = round(
                (desc_config.get("avail_mem") / estimated_memory_usage - 0.22)
                / 0.85
                * self.dim_x
            )
            self._jac_chunk_size = max([1, max_chunk_size])
        if self._deriv_mode == "blocked" and len(self.objectives) > 1:
            # blocked mode should never use this chunk size if there
            # are multiple sub-objectives
            self._jac_chunk_size = None
        elif self._deriv_mode == "blocked" and len(self.objectives) == 1:
            # if there is only one objective i.e. wrapped ForceBalance in
            # ProximalProjection, we can use the chunk size of
            # that objective as if this is batched mode
            self._jac_chunk_size = self.objectives[0]._jac_chunk_size

        if not self.use_jit:
            self._unjit()

        self._built = True

        timer.stop("Objective build")
        if verbose > 1:
            timer.disp("Objective build")

    def _set_things(self, things=None):
        """Tell the ObjectiveFunction what things it is optimizing.

        Parameters
        ----------
        things : list, tuple, or nested list, tuple of Optimizable
            Collection of things used by this objective. Defaults to all things from
            all sub-objectives.

        Notes
        -----
        Sets ``self._flatten`` as a function to return unique flattened list of things
        and ``self._unflatten`` to recreate full nested list of things
        from unique flattened version.

        """
        # This is a unique list of the things the ObjectiveFunction knows about.
        # By default it is only the things that each sub-Objective needs,
        # but it can be set to include extra things from other objectives.
        self._things = setdefault(
            things,
            unique_list(flatten_list([obj.things for obj in self.objectives]))[0],
        )
        things_per_objective = [self._things for _ in self.objectives]

        flat_, treedef_ = tree_flatten(
            things_per_objective, is_leaf=lambda x: isinstance(x, Optimizable)
        )
        unique_, inds_ = unique_list(flat_)

        # this is needed to know which "thing" goes with which sub-objective,
        # ie objectives[i].things == [things[k] for k in things_per_objective_idx[i]]
        self._things_per_objective_idx = []
        for obj in self.objectives:
            self._things_per_objective_idx.append(
                [unique_.index(t) for t in obj.things]
            )

        self._unflatten = _ThingUnflattener(len(unique_), inds_, treedef_)
        self._flatten = _ThingFlattener(len(flat_), treedef_)

    @jit
    def compute_unscaled(self, x, constants=None):
        """Compute the raw value of the objective function.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        params = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        assert len(params) == len(constants) == len(self.objectives)
        f = jnp.concatenate(
            [
                obj.compute_unscaled(*par, constants=const)
                for par, obj, const in zip(params, self.objectives, constants)
            ]
        )
        return f

    @jit
    def compute_scaled(self, x, constants=None):
        """Compute the objective function and apply weighting and normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        params = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        assert len(params) == len(constants) == len(self.objectives)
        f = jnp.concatenate(
            [
                obj.compute_scaled(*par, constants=const)
                for par, obj, const in zip(params, self.objectives, constants)
            ]
        )
        return f

    @jit
    def compute_scaled_error(self, x, constants=None):
        """Compute and apply the target/bounds, weighting, and normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        params = self.unpack_state(x)
        if constants is None:
            constants = self.constants
        assert len(params) == len(constants) == len(self.objectives)
        f = jnp.concatenate(
            [
                obj.compute_scaled_error(*par, constants=const)
                for par, obj, const in zip(params, self.objectives, constants)
            ]
        )
        return f

    @jit
    def compute_scalar(self, x, constants=None):
        """Compute the sum of squares error.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        f : float
            Objective function scalar value.

        """
        f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
        return f

    def print_value(self, x, x0=None, constants=None):
        """Print the value(s) of the objective.

        Parameters
        ----------
        x : ndarray
            State vector.
        x0 : ndarray, optional
            Initial state vector before optimization.
        constants : list
            Constant parameters passed to sub-objectives.

        Returns
        -------
        values: dict
            Dictionary mapping objective titles/names to residual values.
        """
        out = {}
        if constants is None:
            constants = self.constants
        if self.compiled and self._compile_mode in {"scalar", "all"}:
            f = self.compute_scalar(x, constants=constants)
            if x0 is not None:
                f0 = self.compute_scalar(x0, constants=constants)
        else:
            f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
            if x0 is not None:
                f0 = (
                    jnp.sum(self.compute_scaled_error(x0, constants=constants) ** 2) / 2
                )
        if x0 is not None:
            print(
                f"{'Total (sum of squares): ':<{PRINT_WIDTH}}"
                + "{:10.3e}  -->  {:10.3e}, ".format(f0, f)
            )
            temp_out = {"f": f, "f0": f0}
        else:
            print(
                f"{'Total (sum of squares): ':<{PRINT_WIDTH}}" + "{:10.3e}, ".format(f)
            )
            temp_out = {"f": f}
        out["Total (sum of squares)"] = temp_out
        params = self.unpack_state(x)
        assert len(params) == len(constants) == len(self.objectives)
        if x0 is not None:
            params0 = self.unpack_state(x0)
            assert len(params0) == len(constants) == len(self.objectives)
            for par, par0, obj, const in zip(
                params, params0, self.objectives, constants
            ):
                outi = obj.print_value(par, par0, constants=const)
                if obj._print_value_fmt in out:
                    out[obj._print_value_fmt].append(outi)
                else:
                    out[obj._print_value_fmt] = [outi]
        else:
            for par, obj, const in zip(params, self.objectives, constants):
                outi = obj.print_value(par, constants=const)
                if obj._print_value_fmt in out:
                    out[obj._print_value_fmt].append(outi)
                else:
                    out[obj._print_value_fmt] = [outi]
        return out

    def unpack_state(self, x, per_objective=True):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            State vector.
        per_objective : bool
            Whether to return param dicts for each objective (default) or for each
            unique optimizable thing.

        Returns
        -------
        params : pytree of dict
            if per_objective is True, this is a nested list of parameters for each
            sub-Objective, such that self.objectives[i] has parameters params[i].
            Otherwise, it is a list of parameters tied to each optimizable thing
            such that params[i] = self.things[i].params_dict
        """
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")

        x = jnp.atleast_1d(jnp.asarray(x))
        if x.size != self.dim_x:
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self.dim_x} got {x.size}."
            )

        xs_splits = np.cumsum([t.dim_x for t in self.things])
        xs = jnp.split(x, xs_splits)
        xs = xs[: len(self.things)]  # jnp.split returns an empty array at the end
        assert len(xs) == len(self.things)
        params = [t.unpack_params(xi) for t, xi in zip(self.things, xs)]
        if per_objective:
            # params is a list of lists of dicts, for each thing and for each objective
            params = self._unflatten(params)
            # this filters out the params of things that are unused by each objective
            assert len(params) == len(self._things_per_objective_idx)
            params = [
                [param[i] for i in idx]
                for param, idx in zip(params, self._things_per_objective_idx)
            ]
        return params

    def x(self, *things):
        """Return the full state vector from the Optimizable objects things."""
        # TODO (#1392): also check resolution of the things etc?
        things = things or self.things
        errorif(
            len(things) != len(self.things),
            ValueError,
            "Got the wrong number of things, "
            f"expected {len(self.things)} got {len(things)}",
        )
        for t1, t2 in zip(things, self.things):
            errorif(
                not isinstance(t1, type(t2)),
                TypeError,
                f"got incompatible types between things {type(t1)} "
                f"and self.things {type(t2)}",
            )
        xs = [t.pack_params(t.params_dict) for t in things]
        return jnp.concatenate(xs)

    @jit
    def grad(self, x, constants=None):
        """Compute gradient vector of self.compute_scalar wrt x."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_1d(
            Derivative(self.compute_scalar, mode="grad")(x, constants).squeeze()
        )

    @jit
    def hess(self, x, constants=None):
        """Compute Hessian matrix of self.compute_scalar wrt x."""
        if constants is None:
            constants = self.constants
        return jnp.atleast_2d(
            Derivative(self.compute_scalar, mode="hess")(x, constants).squeeze()
        )

    @jit
    def jac_scaled(self, x, constants=None):
        """Compute Jacobian matrix of self.compute_scaled wrt x."""
        v = jnp.eye(x.shape[0])
        return self.jvp_scaled(v, x, constants).T

    @jit
    def jac_scaled_error(self, x, constants=None):
        """Compute Jacobian matrix of self.compute_scaled_error wrt x."""
        v = jnp.eye(x.shape[0])
        return self.jvp_scaled_error(v, x, constants).T

    @jit
    def jac_unscaled(self, x, constants=None):
        """Compute Jacobian matrix of self.compute_unscaled wrt x."""
        v = jnp.eye(x.shape[0])
        return self.jvp_unscaled(v, x, constants).T

    def _jvp_blocked(self, v, x, constants=None, op="scaled"):
        v = ensure_tuple(v)
        if len(v) > 1:
            # using blocked for higher order derivatives is a pain, and only really
            # is needed for perturbations. Just pass that to jvp_batched for now
            return self._jvp_batched(v, x, constants, op)

        if constants is None:
            constants = self.constants
        xs_splits = np.cumsum([t.dim_x for t in self.things])
        xs = jnp.split(x, xs_splits)
        vs = jnp.split(v[0], xs_splits, axis=-1)
        J = []
        assert len(self.objectives) == len(self.constants)
        # basic idea is we compute the jacobian of each objective wrt each thing
        # one by one, and assemble into big block matrix
        # if objective doesn't depend on a given thing, that part is set to 0.
        for k, (obj, const) in enumerate(zip(self.objectives, constants)):
            # get the xs that go to that objective
            thing_idx = self._things_per_objective_idx[k]
            xi = [xs[i] for i in thing_idx]
            vi = [vs[i] for i in thing_idx]
            Ji_ = getattr(obj, "jvp_" + op)(vi, xi, constants=const)
            J += [Ji_]
        # this is the transpose of the jvp when v is a matrix, for consistency with
        # jvp_batched
        J = jnp.hstack(J)
        return J

    def _jvp_batched(self, v, x, constants=None, op="scaled"):
        v = ensure_tuple(v)

        fun = lambda x: getattr(self, "compute_" + op)(x, constants)
        if len(v) == 1:
            jvpfun = lambda dx: Derivative.compute_jvp(fun, 0, dx, x)
            return batched_vectorize(
                jvpfun, signature="(n)->(k)", chunk_size=self._jac_chunk_size
            )(v[0])
        elif len(v) == 2:
            jvpfun = lambda dx1, dx2: Derivative.compute_jvp2(fun, 0, 0, dx1, dx2, x)
            return batched_vectorize(
                jvpfun, signature="(n),(n)->(k)", chunk_size=self._jac_chunk_size
            )(v[0], v[1])
        elif len(v) == 3:
            jvpfun = lambda dx1, dx2, dx3: Derivative.compute_jvp3(
                fun, 0, 0, 0, dx1, dx2, dx3, x
            )
            return batched_vectorize(
                jvpfun,
                signature="(n),(n),(n)->(k)",
                chunk_size=self._jac_chunk_size,
            )(v[0], v[1], v[2])
        else:
            raise NotImplementedError("Cannot compute JVP higher than 3rd order.")

    @jit
    def jvp_scaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            The number of vectors given determines the order of derivative taken.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        if self._deriv_mode == "batched":
            J = self._jvp_batched(v, x, constants, "scaled")
        if self._deriv_mode == "blocked":
            J = self._jvp_blocked(v, x, constants, "scaled")
        return J

    @jit
    def jvp_scaled_error(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            The number of vectors given determines the order of derivative taken.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        if self._deriv_mode == "batched":
            J = self._jvp_batched(v, x, constants, "scaled_error")
        if self._deriv_mode == "blocked":
            J = self._jvp_blocked(v, x, constants, "scaled_error")
        return J

    @jit
    def jvp_unscaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            The number of vectors given determines the order of derivative taken.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        if self._deriv_mode == "batched":
            J = self._jvp_batched(v, x, constants, "unscaled")
        if self._deriv_mode == "blocked":
            J = self._jvp_blocked(v, x, constants, "unscaled")
        return J

    def _vjp(self, v, x, constants=None, op="scaled"):
        fun = lambda x: getattr(self, "compute_" + op)(x, constants)
        return Derivative.compute_vjp(fun, 0, v, x)

    @jit
    def vjp_scaled(self, v, x, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x, constants, "scaled")

    @jit
    def vjp_scaled_error(self, v, x, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled_error.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x, constants, "scaled_error")

    @jit
    def vjp_unscaled(self, v, x, constants=None):
        """Compute vector-Jacobian product of self.compute_unscaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._vjp(v, x, constants, "unscaled")

    def compile(self, mode="auto", verbose=1):
        """Call the necessary functions to ensure the function is compiled.

        Parameters
        ----------
        mode : {"auto", "lsq", "scalar", "bfgs", "all"}
            Whether to compile for least squares optimization or scalar optimization.
            "auto" compiles based on the type of objective, either scalar or lsq
            "bfgs" compiles only scalar objective and gradient,
            "all" compiles all derivatives.
        verbose : int, optional
            Level of output.

        """
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        if not use_jax:
            self._compiled = True
            return

        timer = Timer()
        if mode == "auto" and self.scalar:
            mode = "scalar"
        elif mode == "auto":
            mode = "lsq"
        self._compile_mode = mode
        x = self.x()

        if verbose > 0:
            msg = "Compiling objective function and derivatives: "
            print(msg + f"{[obj.name for obj in self.objectives]}")
        timer.start("Total compilation time")

        if mode in ["scalar", "bfgs", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute_scalar(x, self.constants).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")

            timer.start("Gradient compilation time")
            _ = self.grad(x, self.constants).block_until_ready()
            timer.stop("Gradient compilation time")
            if verbose > 1:
                timer.disp("Gradient compilation time")
        if mode in ["scalar", "all"]:
            timer.start("Hessian compilation time")
            _ = self.hess(x, self.constants).block_until_ready()
            timer.stop("Hessian compilation time")
            if verbose > 1:
                timer.disp("Hessian compilation time")
        if mode in ["lsq", "all"]:
            timer.start("Objective compilation time")
            _ = self.compute_scaled_error(x, self.constants).block_until_ready()
            timer.stop("Objective compilation time")
            if verbose > 1:
                timer.disp("Objective compilation time")

            timer.start("Jacobian compilation time")
            _ = self.jac_scaled_error(x, self.constants).block_until_ready()
            timer.stop("Jacobian compilation time")
            if verbose > 1:
                timer.disp("Jacobian compilation time")

        timer.stop("Total compilation time")
        if verbose > 1:
            timer.disp("Total compilation time")
        self._compiled = True

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        return [obj.constants for obj in self.objectives]

    @property
    def objectives(self):
        """list: List of objectives."""
        return self._objectives

    @property
    def use_jit(self):
        """bool: Whether to just-in-time compile the objective and derivatives."""
        return self._use_jit

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar or vector."""
        if not self._built:
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._scalar

    @property
    def built(self):
        """bool: Whether the objectives have been built or not."""
        return self._built

    @property
    def compiled(self):
        """bool: Whether the functions have been compiled or not."""
        return self._compiled

    @property
    def dim_x(self):
        """int: Dimensional of the state vector."""
        return sum(t.dim_x for t in self.things)

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        if not hasattr(self, "_dim_f"):
            raise RuntimeError("ObjectiveFunction must be built first.")
        return self._dim_f

    @property
    def name(self):
        """Name of objective function (str)."""
        return self.__dict__.setdefault("_name", "")

    @property
    def target_scaled(self):
        """ndarray: target vector."""
        target = []
        for obj in self.objectives:
            if obj.target is not None:
                target_i = jnp.ones(obj.dim_f) * obj.target
            else:
                # need to return something, so use midpoint of bounds as approx target
                target_i = jnp.ones(obj.dim_f) * (obj.bounds[0] + obj.bounds[1]) / 2
            target_i = obj._scale(target_i)
            if not obj._normalize_target:
                target_i *= obj.normalization
            target += [target_i]
        return jnp.concatenate(target)

    @property
    def bounds_scaled(self):
        """tuple: lower and upper bounds for residual vector."""
        lb, ub = [], []
        for obj in self.objectives:
            if obj.bounds is not None:
                lb_i = jnp.ones(obj.dim_f) * obj.bounds[0]
                ub_i = jnp.ones(obj.dim_f) * obj.bounds[1]
            else:
                lb_i = jnp.ones(obj.dim_f) * obj.target
                ub_i = jnp.ones(obj.dim_f) * obj.target
            lb_i = obj._scale(lb_i)
            ub_i = obj._scale(ub_i)
            if not obj._normalize_target:
                lb_i *= obj.normalization
                ub_i *= obj.normalization
            lb += [lb_i]
            ub += [ub_i]
        return (jnp.concatenate(lb), jnp.concatenate(ub))

    @property
    def weights(self):
        """ndarray: weight vector."""
        return jnp.concatenate(
            [jnp.ones(obj.dim_f) * obj.weight for obj in self.objectives]
        )

    @property
    def things(self):
        """list: Unique list of optimizable things that this objective is tied to."""
        return self._things


class _Objective(IOAble, ABC):
    """Objective (or constraint) used in the optimization of an Equilibrium.

    Parameters
    ----------
    things : Optimizable or tuple/list of Optimizable
        Objects that will be optimized to satisfy the Objective."""  # noqa: D208, D209

    _scalar = False
    _linear = False
    _coordinates = ""
    _units = "(Unknown)"
    _equilibrium = False
    _io_attrs_ = [
        "_bounds",
        "_deriv_mode",
        "_name",
        "_normalize",
        "_normalize_target",
        "_normalization",
        "_target",
        "_weight",
    ]
    _static_attrs = [
        "_built",
        "_coordinates",
        "_data_keys",
        "_deriv_mode",
        "_dim_f",
        "_equilibrium",
        "_jac_chunk_size",
        "_linear",
        "_loss_function",
        "_name",
        "_normalize",
        "_normalize_target",
        "_print_value_fmt",
        "_scalar",
        "_units",
        "_static_attrs",
    ]

    def __init__(
        self,
        things=None,
        target=0,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        name=None,
        jac_chunk_size=None,
    ):
        if self._scalar:
            assert self._coordinates == ""
        assert np.all(np.asarray(weight) > 0)
        assert normalize in {True, False}
        assert normalize_target in {True, False}
        assert (bounds is None) or (isinstance(bounds, tuple) and len(bounds) == 2)
        assert (bounds is None) or (target is None), "Cannot use both bounds and target"
        assert loss_function in [None, "mean", "min", "max", "sum"]
        assert deriv_mode in {"auto", "fwd", "rev"}
        assert jac_chunk_size is None or isposint(jac_chunk_size)

        self._jac_chunk_size = jac_chunk_size

        self._target = target
        self._bounds = bounds
        self._weight = weight
        self._normalize = normalize
        self._normalize_target = normalize_target
        self._normalization = 1
        self._deriv_mode = deriv_mode
        self._name = name
        self._use_jit = True
        self._built = False
        self._loss_function = {
            "mean": jnp.mean,
            "max": jnp.max,
            "min": jnp.min,
            "sum": jnp.sum,
            None: None,
        }[loss_function]

        self._things = flatten_list([things], True)

    def _set_derivatives(self):
        """Choose derivative mode based on size of inputs/outputs."""
        if self._deriv_mode == "auto":
            # choose based on shape of jacobian. dim_x is usually an overestimate of
            # the true number of DOFs because linear constraints remove some. Also
            # fwd mode is more memory efficient so we prefer that unless the jacobian
            # is really wide
            self._deriv_mode = (
                "fwd"
                if self.dim_f >= 0.2 * sum(t.dim_x for t in self.things)
                else "rev"
            )

    def _unjit(self):
        """Remove jit compiled methods."""
        methods = [
            "compute_scaled",
            "compute_scaled_error",
            "compute_unscaled",
            "compute_scalar",
            "jac_scaled",
            "jac_scaled_error",
            "jac_unscaled",
            "jvp_scaled",
            "jvp_scaled_error",
            "jvp_unscaled",
            "hess",
            "grad",
        ]
        for method in methods:
            try:
                setattr(
                    self, method, functools.partial(getattr(self, method)._fun, self)
                )
                if method not in self._static_attrs:
                    self._static_attrs += [method]
            except AttributeError:
                pass

    def _check_dimensions(self):
        """Check that len(target) = len(bounds) = len(weight) = dim_f."""
        if self.bounds is not None:  # must be a tuple of length 2
            self._bounds = tuple([np.asarray(bound) for bound in self._bounds])
            for bound in self.bounds:
                if not is_broadcastable((self.dim_f,), bound.shape) or (
                    self.dim_f == 1 and bound.size != 1
                ):
                    raise ValueError("len(bounds) != dim_f")
            if np.any(self.bounds[1] < self.bounds[0]):
                raise ValueError("bounds must be: (lower bound, upper bound)")
        else:  # target only gets used if bounds is None
            self._target = np.asarray(self._target)
            if not is_broadcastable((self.dim_f,), self.target.shape) or (
                self.dim_f == 1 and self.target.size != 1
            ):
                raise ValueError("len(target) != dim_f")

        self._weight = np.asarray(self._weight)
        if not is_broadcastable((self.dim_f,), self.weight.shape) or (
            self.dim_f == 1 and self.weight.size != 1
        ):
            raise ValueError("len(weight) != dim_f")

    @abstractmethod
    def build(self, use_jit=True, verbose=1):
        """Build constant arrays."""
        self._check_dimensions()
        self._set_derivatives()

        # set quadrature weights if they haven't been
        if hasattr(self, "_constants") and ("quad_weights" not in self._constants):
            grid = self._constants["transforms"]["grid"]
            if self._coordinates == "rtz":
                w = grid.weights
                w *= jnp.sqrt(grid.num_nodes)
            elif self._coordinates == "r":
                w = grid.compress(grid.spacing[:, 0], surface_label="rho")
                w = jnp.sqrt(w)
            else:
                w = jnp.ones((self.dim_f,))
            if w.size:
                w = jnp.tile(w, self.dim_f // w.size)
            self._constants["quad_weights"] = w

        if self._loss_function is not None:
            self._dim_f = 1
            if hasattr(self, "_constants"):
                self._constants["quad_weights"] = 1.0

        if use_jit is not None:
            self._use_jit = use_jit
        if not self._use_jit:
            self._unjit()

        self._built = True

    @abstractmethod
    def compute(self, *args, **kwargs):
        """Compute the objective function."""

    def _maybe_array_to_params(self, *args):
        argsout = tuple()
        assert len(args) == len(self.things)
        for arg, thing in zip(args, self.things):
            if isinstance(arg, (np.ndarray, jnp.ndarray)):
                argsout += (thing.unpack_params(arg),)
            else:
                argsout += (arg,)
        return argsout

    @jit
    def compute_unscaled(self, *args, **kwargs):
        """Compute the raw value of the objective."""
        args = self._maybe_array_to_params(*args)
        f = self.compute(*args, **kwargs)
        if self._loss_function is not None:
            f = self._loss_function(f)
        return jnp.atleast_1d(f)

    @jit
    def compute_scaled(self, *args, **kwargs):
        """Compute and apply weighting and normalization."""
        args = self._maybe_array_to_params(*args)
        f = self.compute(*args, **kwargs)
        if self._loss_function is not None:
            f = self._loss_function(f)
        return jnp.atleast_1d(self._scale(f, **kwargs))

    @jit
    def compute_scaled_error(self, *args, **kwargs):
        """Compute and apply the target/bounds, weighting, and normalization."""
        args = self._maybe_array_to_params(*args)
        f = self.compute(*args, **kwargs)
        if self._loss_function is not None:
            f = self._loss_function(f)
        return jnp.atleast_1d(self._scale(self._shift(f), **kwargs))

    def _shift(self, f):
        """Subtract target or clamp to bounds."""
        if self.bounds is not None:  # using lower/upper bounds instead of target
            if self._normalize_target:
                bounds = self.bounds
            else:
                bounds = tuple([bound * self.normalization for bound in self.bounds])
            f_target = jnp.where(  # where f is within target bounds, return 0 error
                jnp.logical_and(f >= bounds[0], f <= bounds[1]),
                jnp.zeros_like(f),
                jnp.where(  # otherwise return error = f - bound
                    jnp.abs(f - bounds[0]) < jnp.abs(f - bounds[1]),
                    f - bounds[0],  # errors below lower bound are negative
                    f - bounds[1],  # errors above upper bound are positive
                ),
            )
        else:  # using target instead of lower/upper bounds
            if self._normalize_target:
                target = self.target
            else:
                target = self.target * self.normalization
            f_target = f - target
        return f_target

    def _scale(self, f, *args, **kwargs):
        """Apply weighting, normalization etc."""
        constants = kwargs.get("constants", self.constants)
        if constants is None:
            w = jnp.ones_like(f)
        else:
            w = constants["quad_weights"]
        f_norm = jnp.atleast_1d(f) / self.normalization  # normalization
        return f_norm * w * self.weight

    @jit
    def compute_scalar(self, *args, **kwargs):
        """Compute the scalar form of the objective."""
        if self.scalar:
            f = self.compute_scaled_error(*args, **kwargs)
        else:
            f = jnp.sum(self.compute_scaled_error(*args, **kwargs) ** 2) / 2
        return f.squeeze()

    @jit
    def grad(self, *args, **kwargs):
        """Compute gradient vector of self.compute_scalar wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(self.compute_scalar, argnums, mode="grad")(*args, **kwargs)

    @jit
    def hess(self, *args, **kwargs):
        """Compute Hessian matrix of self.compute_scalar wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(self.compute_scalar, argnums, mode="hess")(*args, **kwargs)

    @jit
    def jac_scaled(self, *args, **kwargs):
        """Compute Jacobian matrix of self.compute_scaled wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(
            self.compute_scaled,
            argnums,
            mode=self._deriv_mode,
            chunk_size=self._jac_chunk_size,
        )(*args, **kwargs)

    @jit
    def jac_scaled_error(self, *args, **kwargs):
        """Compute Jacobian matrix of self.compute_scaled_error wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(
            self.compute_scaled_error,
            argnums,
            mode=self._deriv_mode,
            chunk_size=self._jac_chunk_size,
        )(*args, **kwargs)

    @jit
    def jac_unscaled(self, *args, **kwargs):
        """Compute Jacobian matrix of self.compute_unscaled wrt x."""
        argnums = tuple(range(len(self.things)))
        return Derivative(
            self.compute_unscaled,
            argnums,
            mode=self._deriv_mode,
            chunk_size=self._jac_chunk_size,
        )(*args, **kwargs)

    def _jvp(self, v, x, constants=None, op="scaled"):
        v = ensure_tuple(v)
        x = ensure_tuple(x)
        assert len(x) == len(v)

        if self._deriv_mode == "fwd":
            fun = lambda *x: getattr(self, "compute_" + op)(*x, constants=constants)
            jvpfun = lambda *dx: Derivative.compute_jvp(
                fun, tuple(range(len(x))), dx, *x
            )
            sig = ",".join(f"(n{i})" for i in range(len(x))) + "->(k)"
            return batched_vectorize(
                jvpfun, signature=sig, chunk_size=self._jac_chunk_size
            )(*v)
        else:  # rev mode. We compute full jacobian and manually do mv. In this case
            # the jacobian should be wide so this isn't very expensive.
            jac = getattr(self, "jac_" + op)(*x, constants=constants)
            # jac is a tuple, 1 array for each thing. Transposes here and below make it
            # equivalent to fwd mode above, which batches over the first axis
            Jv = tree_map(lambda a, b: jnp.dot(a, b.T), jac, v)
            # sum over different things.
            return jnp.sum(jnp.asarray(Jv), axis=0).T

    @jit
    def jvp_scaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x : tuple of ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x, constants, "scaled")

    @jit
    def jvp_scaled_error(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x : tuple of ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x, constants, "scaled_error")

    @jit
    def jvp_unscaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x : tuple of ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives.

        """
        return self._jvp(v, x, constants, "unscaled")

    def print_value(self, args, args0=None, **kwargs):  # noqa: C901
        """Print the value of the objective and return a dict of values."""
        # compute_unscaled is jitted so better to use than than bare compute
        out = {}
        if args0 is not None:
            f = self.compute_unscaled(*args, **kwargs)
            f0 = self.compute_unscaled(*args0, **kwargs)
            print_value_fmt = (
                f"{self._print_value_fmt:<{PRINT_WIDTH}}" + "{:10.3e}  -->  {:10.3e} "
            )
        else:
            f = self.compute_unscaled(*args, **kwargs)
            f0 = f
            # In this case, print_value_fmt only has 1 value,
            # but the format string is still used with 2 arguments given.
            # This is a bit of a hack, but it works. the format() only replaces
            # the first value in the {} string, so the second one is unused.
            # That is why we set f0 to f.
            print_value_fmt = f"{self._print_value_fmt:<{PRINT_WIDTH}}" + "{:10.3e} "

        if self.linear:
            # probably a Fixed* thing, just need to know norm
            f = jnp.linalg.norm(self._shift(f))
            f0 = jnp.linalg.norm(self._shift(f0))
            print(print_value_fmt.format(f0, f) + self._units)
            out["f"] = f
            if args0 is not None:
                out["f0"] = f0

        elif self.scalar:
            # dont need min/max/mean of a scalar
            fs = f.squeeze()
            f0s = f0.squeeze()
            print(print_value_fmt.format(f0s, fs) + self._units)
            out["f"] = fs
            if args0 is not None:
                out["f0"] = f0s
            if self._normalize and self._units != "(dimensionless)":
                fs_norm = self._scale(self._shift(f)).squeeze()
                f0s_norm = self._scale(self._shift(f0)).squeeze()
                print(print_value_fmt.format(f0s_norm, fs_norm) + "(normalized error)")
                out["f_norm"] = fs_norm
                if args0 is not None:
                    out["f0_norm"] = f0s_norm

        else:
            # try to do weighted mean if possible
            constants = kwargs.get("constants", self.constants)
            if constants is None:
                w = jnp.ones_like(f)
            else:
                w = constants["quad_weights"]

            # target == 0 probably indicates f is some sort of error metric,
            # mean abs makes more sense than mean
            abserr = jnp.all(self.target == 0)
            f = jnp.abs(f) if abserr else f
            fmax = jnp.max(f)
            fmin = jnp.min(f)
            fmean = jnp.mean(f * w) / jnp.mean(w)

            f0 = jnp.abs(f0) if abserr else f0
            f0max = jnp.max(f0)
            f0min = jnp.min(f0)
            f0mean = jnp.mean(f0 * w) / jnp.mean(w)

            pre_width = len("Maximum absolute ") if abserr else len("Maximum ")
            if args0 is not None:
                print_value_fmt = (
                    f"{self._print_value_fmt:<{PRINT_WIDTH-pre_width}}"
                    + "{:10.3e}  -->  {:10.3e} "
                )
            else:
                print_value_fmt = (
                    f"{self._print_value_fmt:<{PRINT_WIDTH-pre_width}}" + "{:10.3e} "
                )
            print(
                "Maximum "
                + ("absolute " if abserr else "")
                + print_value_fmt.format(f0max, fmax)
                + self._units
            )
            out["f_max"] = fmax
            if args0 is not None:
                out["f0_max"] = f0max
            print(
                "Minimum "
                + ("absolute " if abserr else "")
                + print_value_fmt.format(f0min, fmin)
                + self._units
            )
            out["f_min"] = fmin
            if args0 is not None:
                out["f0_min"] = f0min
            print(
                "Average "
                + ("absolute " if abserr else "")
                + print_value_fmt.format(f0mean, fmean)
                + self._units
            )
            out["f_mean"] = fmean
            if args0 is not None:
                out["f0_mean"] = f0mean

            if self._normalize and self._units != "(dimensionless)":
                fmax_norm = fmax / jnp.mean(self.normalization)
                fmin_norm = fmin / jnp.mean(self.normalization)
                fmean_norm = fmean / jnp.mean(self.normalization)

                f0max_norm = f0max / jnp.mean(self.normalization)
                f0min_norm = f0min / jnp.mean(self.normalization)
                f0mean_norm = f0mean / jnp.mean(self.normalization)

                print(
                    "Maximum "
                    + ("absolute " if abserr else "")
                    + print_value_fmt.format(f0max_norm, fmax_norm)
                    + "(normalized)"
                )
                out["f_max_norm"] = fmax_norm
                if args0 is not None:
                    out["f0_max_norm"] = f0max_norm
                print(
                    "Minimum "
                    + ("absolute " if abserr else "")
                    + print_value_fmt.format(f0min_norm, fmin_norm)
                    + "(normalized)"
                )
                out["f_min_norm"] = fmin_norm
                if args0 is not None:
                    out["f0_min_norm"] = f0min_norm
                print(
                    "Average "
                    + ("absolute " if abserr else "")
                    + print_value_fmt.format(f0mean_norm, fmean_norm)
                    + "(normalized)"
                )
                out["f_mean_norm"] = fmean_norm
                if args0 is not None:
                    out["f0_mean_norm"] = f0mean_norm
        return out

    def xs(self, *things):
        """Return a tuple of args required by this objective from optimizable things."""
        things = things or self.things
        errorif(
            len(things) != len(self.things),
            ValueError,
            "Got the wrong number of things, "
            f"expected {len(self.things)} got {len(things)}",
        )
        for t1, t2 in zip(things, self.things):
            errorif(
                not isinstance(t1, type(t2)),
                TypeError,
                f"got incompatible types between things {type(t1)} "
                f"and self.things {type(t2)}",
            )
        return tuple([t.params_dict for t in things])

    @property
    def constants(self):
        """dict: Constant parameters such as transforms and profiles."""
        if hasattr(self, "_constants"):
            return self._constants
        return None

    @property
    def target(self):
        """float: Target value(s) of the objective."""
        return self._target

    @target.setter
    def target(self, target):
        self._target = np.atleast_1d(target) if target is not None else target
        self._check_dimensions()

    @property
    def bounds(self):
        """tuple: Lower and upper bounds of the objective."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        assert (bounds is None) or (isinstance(bounds, tuple) and len(bounds) == 2)
        self._bounds = bounds
        self._check_dimensions()

    @property
    def weight(self):
        """float: Weighting to apply to the Objective, relative to other Objectives."""
        return self._weight

    @weight.setter
    def weight(self, weight):
        assert np.all(np.asarray(weight) > 0)
        self._weight = np.atleast_1d(weight)
        self._check_dimensions()

    @property
    def normalization(self):
        """float: normalizing scale factor."""
        if self._normalize and not self.built:
            raise ValueError("Objective must be built first")
        return self._normalization

    @property
    def built(self):
        """bool: Whether the transforms have been precomputed (or not)."""
        return self._built

    @property
    def dim_f(self):
        """int: Number of objective equations."""
        return self._dim_f

    @property
    def scalar(self):
        """bool: Whether default "compute" method is a scalar or vector."""
        return self._scalar

    @property
    def linear(self):
        """bool: Whether the objective is a linear function (or nonlinear)."""
        return self._linear

    @property
    def fixed(self):
        """bool: Whether the objective fixes individual parameters (or linear combo)."""
        if self.linear:
            return self._fixed
        else:
            return False

    @property
    def name(self):
        """Name of objective (str)."""
        return self.__dict__.setdefault("_name", "")

    @property
    def things(self):
        """list: Optimizable things that this objective is tied to."""
        if not hasattr(self, "_things"):
            self._things = []
        return list(self._things)

    @things.setter
    def things(self, new):
        if not isinstance(new, (tuple, list)):
            new = [new]
        assert all(isinstance(x, Optimizable) for x in new)
        assert len(new) == len(self.things)
        assert all(type(a) is type(b) for a, b in zip(new, self.things))
        self._things = list(new)
        # can maybe improve this later to not rebuild if resolution is the same
        self._built = False


_Objective.__doc__ += "".join(value.rstrip("\n") for value in docs.values())

# local functions assigned as attributes aren't hashable so they cause stuff to
# recompile, so instead we define a hashable class to do the same thing.


class _ThingUnflattener(IOAble):

    _static_attrs = ["length", "inds", "treedef"]

    def __init__(self, length, inds, treedef):
        self.length = length
        self.inds = inds
        self.treedef = treedef

    def __call__(self, unique):
        assert len(unique) == self.length
        flat = [unique[i] for i in self.inds]
        return tree_unflatten(self.treedef, flat)


class _ThingFlattener(IOAble):

    _static_attrs = ["length", "treedef"]

    def __init__(self, length, treedef):
        self.length = length
        self.treedef = treedef

    def __call__(self, things):
        flat, treedef = tree_flatten(
            things, is_leaf=lambda x: isinstance(x, Optimizable)
        )
        assert treedef == self.treedef
        assert len(flat) == self.length
        unique, _ = unique_list(flat)
        return unique
