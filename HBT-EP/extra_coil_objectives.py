import numpy as np

from desc.backend import jnp
from desc.backend import tree_leaves
from desc.compute import get_profiles, get_transforms
from desc.compute.utils import _compute as compute_fun
from desc.grid import LinearGrid
from desc.utils import Timer

from desc.objectives.normalization import compute_scaling_factors
from desc.objectives.objective_funs import _Objective, collect_docs
from desc.objectives._coils import _CoilObjective


class CoilBounds(_CoilObjective):
    """Target the coil to stay inside a torus.

    Parameters
    ----------
    coil : CoilSet or Coil
        Coil(s) that are to be optimized
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``LinearGrid(N=2 * coil.N + 5)``

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=2*np.pi``.",
        bounds_default="``target=2*np.pi``.",
        coil=True,
    )

    _scalar = False  # Not always a scalar, if a coilset is passed in
    _units = "(m)"
    _print_value_fmt = "Coil bound: "
    _broadcast_input = "Coil"

    def __init__(
        self,
        coil,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="coil bound",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 2 * np.pi

        super().__init__(
            coil,
            ["R", "Z"],
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            grid=grid,
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
        super().build(use_jit=use_jit, verbose=verbose)

        self._constants["quad_weights"] = 1

        if self._normalize:
            self._normalization = np.mean([scale["a"] for scale in self._scales])

        _Objective.build(self, use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute coil length.

        Parameters
        ----------
        params : dict
            Dictionary of the coil's degrees of freedom.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self._constants.

        Returns
        -------
        f : array of floats
            Coil length.

        """
        data = super().compute(params, constants=constants)
        data = tree_leaves(data, is_leaf=lambda x: isinstance(x, dict))
        # out = jnp.array([dat["R"]  for dat in data])

        out = jnp.array(
            [
                jnp.sum(jnp.maximum(0.75 - dat["R"], 0) ** 2)
                + jnp.sum(jnp.maximum(dat["R"] - 1.25, 0) ** 2)
                + jnp.sum(jnp.maximum(-0.25 - dat["Z"], 0) ** 2)
                + jnp.sum(jnp.maximum(dat["Z"] - 0.25, 0) ** 2)
                for dat in data
            ]
        )

        return out[self._coilset_tree["coilset_mask"]]


class SurfaceMatch(_Objective):
    """Target a surface shape.

    Try to match a surface with another surface shape

    Parameters
    ----------
    surface : FourierRZToroidalSurface
        QFM surface upon which the normal field error will be minimized.
    surfacet : MagneticField
        External field produced by coils or other source, which will be optimized to
        minimize the normal field error on the provided QFM surface. May be fixed
        by passing in ``field_fixed=True``
    eval_grid : Grid, optional
        Collocation grid containing the nodes on the surface at which the
        magnetic field is being calculated and where to evaluate Bn errors.
        Default grid is: ``LinearGrid(rho=np.array([1.0]), M=surface.M_grid,``
        ``N=surface.N_grid, NFP=surface.NFP, sym=False)``
    field_grid : Grid, optional
        Grid used to discretize field (e.g. grid for the magnetic field source from
        coils). Default grid is determined by the specific MagneticField object, see
        the docs of that object's ``compute_magnetic_field`` method for more detail.
    field_fixed : bool
        Whether or not to fix the magnetic field's DOFs during the optimization.
    bs_chunk_size : int or None
        Size to split Biot-Savart computation into chunks of evaluation points.
        If no chunking should be done or the chunk size is the full input
        then supply ``None``.

    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.",
        bounds_default="``target=0``.",
    )

    _scalar = False
    _linear = False
    _print_value_fmt = "Surfaces match: "
    _units = "(T m^2)"
    _coordinates = "rtz"

    def __init__(
        self,
        surface,
        # field,
        surfacet,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        surfacet_grid=None,
        name="Surface Match",
        surfacet_fixed=True,
        *,
        jac_chunk_size=None,
        # bs_chunk_size=None,
        **kwargs,
    ):
        if target is None and bounds is None:
            target = 0
        self._surface = surface
        # self._field = field
        self._surfacet = surfacet
        self._surfacet_fixed = surfacet_fixed
        self._surfacet_grid = surfacet_grid
        # self._bs_chunk_size = bs_chunk_size

        things = [surface]
        if not surfacet_fixed:
            things += [surfacet]
        super().__init__(
            things=things,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
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
        surfacet = self._surfacet

        if self._surfacet_grid is None:
            surfacet_grid = LinearGrid(
                rho=np.array([1.0]),
                M=2 * surfacet.M,
                N=2 * surfacet.N,
                NFP=surfacet.NFP,
                sym=False,
            )
            self._surfacet_grid = surfacet_grid
        else:
            surfacet_grid = self._surfacet_grid

        self._data_keys = ["R", "Z", "phi"]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        self._dim_f = surfacet_grid.num_nodes

        w = surfacet_grid.weights
        w *= jnp.sqrt(surfacet_grid.num_nodes)

        surfacet_profiles = get_profiles(
            self._data_keys, obj=surfacet, grid=surfacet_grid
        )
        surfacet_transforms = get_transforms(
            self._data_keys, obj=surfacet, grid=surfacet_grid
        )

        # Target surface data is calculate in build
        # because that's not going to change
        surfacet_data = compute_fun(
            surfacet,
            self._data_keys,
            params=surfacet.params_dict,
            transforms=surfacet_transforms,
            profiles=surfacet_profiles,
        )

        self._constants = {
            "surfacet": self._surfacet,
            "surfacet_grid": self._surfacet_grid,
            "quad_weights": w,
            "surfacet_data": surfacet_data,
            "surfacet_transforms": surfacet_transforms,
            "surfacet_profiles": surfacet_profiles,
        }

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(surfacet)
            Bscale = 1.0  # surface has no inherent B scale
            self._normalization = Bscale * scales["R0"] * scales["a"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params_1, params_2=None, constants=None):
        """Compute normal field on surface.

        Parameters
        ----------
        params_1 : dict
            Dictionary of the surface's degrees of freedom.
        params_2 : dict
            Dictionary of the external field's degrees of freedom, only provided if
            if field_fixed=False.
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            Bnorm on the QFM surface from the external field

        """
        if constants is None:
            constants = self.constants
        # surft_params = params_2 if not self._surfacet_fixed else None
        surft_params = None
        surf_params = params_1

        # surface data is calculated in compute because that will change
        surface_data = compute_fun(
            self._surface,
            self._data_keys,
            surf_params,
            constants["surfacet_transforms"],
            constants["surfacet_profiles"],
        )
        xR = surface_data["R"]
        xZ = surface_data["Z"]

        # if surfacet_params is None:
        #    surfacet_params = constants["surfacet"].params_dict

        y = constants["surfacet_data"]
        yR = y["R"]
        yZ = y["Z"]

        f = jnp.sqrt((xR - yR) ** 2 + (xZ - yZ) ** 2)
        # f = xR**2 + xZ**2
        # f = yR**2 + yZ**2
        return f.ravel()
