import jax
import jax.numpy as jnp
from desc.grid import LinearGrid
from desc.equilibrium import Equilibrium


def transform_matrix(eq, nodes):
    """matrix found by some FFTs and other operations"""
    # find some matrix A that evaluates nodes on Fourier-Zernike basis
    return


class Objective:
    def __init__(self, eq, nodes):
        """Objective function for optimization.

        Parameters
        ----------
        eq : Equilibrium object
            Equilibrium object that we will optimize. This has the information
            about the Fourier-Zernike basis coefficients.
        nodes : jnp.ndarray or some Grid Object
            The nodes at which we want to evaluate the objective function value.
            You can think these as the R, Z and Phi in cylindrical coordinates but
            we actually use some other coordinate system.
        """
        self.nodes = nodes
        self.A = transform_matrix(eq, nodes)
        # in reality this is a much more complicated function

    @jax.jit  # non-parallelized version is jittable
    def compute(self, params):
        """Compute the objective function.

        Parameters
        ----------
        params : jnp.ndarray
            The Fourier-Zernike basis coefficients that we will optimize.

        Returns
        -------
        jnp.ndarray
            Objective function value.
        """
        # here we do some operations with self.A and params to compute the objective
        # function value. Apart from the transform matrix `self.A`, the computation
        # is independent for some nodes.
        # Computation is only dependent on the nodes that have the same R coordinate.
        # So, in theory, we should be able to parallelize the computation for each R.
        return


class ObjectiveFunction:
    def __init__(self, objectives):
        """Final Objective function for optimization.

        Parameters
        ----------
        objectives : list of Objective objects
            Objective objects that we will optimize.
        """
        self.objectives = objectives

    @jax.jit  # non-parallelized version is jittable
    def compute(self, params):
        """Compute the objective function.

        Parameters
        ----------
        params : jnp.ndarray
            The Fourier-Zernike basis coefficients that we will optimize.

        Returns
        -------
        f : jnp.ndarray
            Objective function value.
        """
        f = jnp.concatenate([obj.compute(params) for obj in self.objectives])
        return f


# Create some objects
nodes = [LinearGrid() for _ in range(10)]
eq = Equilibrium()
# In general each objective is different but with the same Equilibrium object
objectives = ObjectiveFunction([Objective(eq, nodes) for node in nodes])

obj_value = objectives.compute(eq.params_dict)
# take the Jacobian using auto-differentiation, in actual code we use jvp and vjps
Jacobian = jax.grad(objectives.compute)(eq.params_dict)
