# mypy: allow-untyped-defs
"""Physics-Informed Neural Networks (PINNs).

Implements the collocation-based PINN framework introduced in Raissi et al. (2019)
and reviewed comprehensively in Cuomo et al., J. Sci. Comput. 92:88 (2022).

A PINN approximates the solution u(z) of a PDE system:

    F(u(z); gamma) = f(z),   z in Omega
    B(u(z))        = g(z),   z in dOmega

by training a neural network u_theta to minimise the combined residual loss:

    L(theta) = w_F * L_F(theta) + w_B * L_B(theta) + w_d * L_data(theta)

where
    L_F    = (1/N_c) sum_i ||F(u_theta(z_i)) - f(z_i)||^2  (PDE residual at collocation pts)
    L_B    = (1/N_b) sum_i ||B(u_theta(z_i)) - g(z_i)||^2  (boundary / initial conditions)
    L_data = (1/N_d) sum_i ||u_theta(z_i) - u*_i||^2        (optional observed data)

Derivatives in F are computed via torch.autograd (automatic differentiation).

Classes
-------
PINN
    Full PINN: backbone network + weighted multi-term loss + gradient API.

PINNMixin
    Lightweight mixin that adds ``pde_residual()``, ``bc_residual()``, and
    ``pinn_loss()`` to any existing ``nn.Module`` backbone.

Helpers
-------
pinn_grad        Compute d^k u / dz_i^k via autograd (creates graph).
pinn_collocation Uniformly sample collocation points in a box domain.
"""

from collections.abc import Callable, Sequence
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .activation import Tanh
from .container import Sequential
from .dropout import Dropout
from .linear import Linear
from .module import Module
from .normalization import LayerNorm


__all__ = [
    "PINN",
    "PINNMixin",
    "pinn_grad",
    "pinn_collocation",
]


# ---------------------------------------------------------------------------
# Helper: automatic differentiation derivative
# ---------------------------------------------------------------------------

def pinn_grad(
    u: Tensor,
    z: Tensor,
    *,
    order: int = 1,
    create_graph: bool = True,
) -> Tensor:
    """Compute the ``order``-th derivative of ``u`` w.r.t. ``z`` via autograd.

    Both ``u`` and ``z`` must have been produced inside a ``torch.enable_grad``
    context and ``z`` must have ``requires_grad=True``.

    Args:
        u: network output, shape ``(...,)`` or ``(..., out)``; must be scalar
           or the gradient is taken element-wise over the last dimension via
           ``torch.sum`` to obtain a scalar, then differentiated.
        z: input tensor with ``requires_grad=True``.
        order: derivative order (default 1). For order > 1 the function calls
               itself recursively.
        create_graph: if ``True`` (default), the gradient computation is added
                      to the computational graph, enabling higher-order
                      derivatives and backprop through the residual loss.

    Returns:
        Tensor of the same shape as ``z``.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    # sum to scalar if u has more than one element (keeps shapes aligned)
    u_sum = u.sum()
    (grad,) = torch.autograd.grad(
        u_sum,
        z,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )
    if grad is None:
        return torch.zeros_like(z)

    if order == 1:
        return grad
    return pinn_grad(grad, z, order=order - 1, create_graph=create_graph)


# ---------------------------------------------------------------------------
# Helper: collocation point sampler
# ---------------------------------------------------------------------------

def pinn_collocation(
    n: int,
    domain: Sequence[tuple[float, float]],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = True,
) -> Tensor:
    """Uniformly sample ``n`` collocation points from a box domain.

    Args:
        n: number of collocation points.
        domain: sequence of ``(lo, hi)`` pairs, one per spatial/temporal
                dimension, e.g. ``[(0, 1), (0, 1)]`` for a 2-D unit square.
        device: target device (defaults to CPU).
        dtype: floating-point dtype.
        requires_grad: if ``True`` (default), the returned tensor tracks
                       gradients so that ``pinn_grad`` can be applied.

    Returns:
        Tensor of shape ``(n, d)`` where ``d = len(domain)``.
    """
    d = len(domain)
    lo = torch.tensor([b[0] for b in domain], dtype=dtype, device=device)
    hi = torch.tensor([b[1] for b in domain], dtype=dtype, device=device)
    pts = torch.rand(n, d, dtype=dtype, device=device)
    pts = pts * (hi - lo) + lo
    pts.requires_grad_(requires_grad)
    return pts


# ---------------------------------------------------------------------------
# PINNMixin: add PINN loss terms to any backbone
# ---------------------------------------------------------------------------

class PINNMixin:
    """Mixin that adds physics-informed loss helpers to any ``nn.Module``.

    Subclass alongside ``nn.Module``::

        class MyPINN(PINNMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(2, 1)

            def forward(self, z):
                return self.net(z)

            def pde_operator(self, u, z):
                # e.g. Poisson: -u_xx - u_yy - f(z)
                u_xx = pinn_grad(pinn_grad(u, z)[:, 0], z)[:, 0]
                u_yy = pinn_grad(pinn_grad(u, z)[:, 1], z)[:, 1]
                return -u_xx - u_yy

    ``pde_operator`` and ``bc_operator`` are abstract -- subclasses must
    override the ones they need.
    """

    def pde_operator(
        self,
        u: Tensor,
        z: Tensor,
    ) -> Tensor:
        """Return the PDE residual F(u_theta(z)) - f(z).

        Override in subclasses.  The default returns zeros (trivially satisfied).
        """
        return torch.zeros_like(u)

    def bc_operator(
        self,
        u: Tensor,
        z_bc: Tensor,
        g: Tensor,
    ) -> Tensor:
        """Return the boundary-condition residual B(u_theta(z_bc)) - g(z_bc).

        Default: Dirichlet  B(u) - g = u - g.
        """
        return u - g

    def pde_residual(self, z_col: Tensor) -> Tensor:
        """Evaluate the PDE residual on collocation points ``z_col``.

        Args:
            z_col: ``(N_c, d)`` collocation points with ``requires_grad=True``.

        Returns:
            ``(N_c, out)`` residual tensor.
        """
        u = self(z_col)  # type: ignore[operator]
        return self.pde_operator(u, z_col)

    def bc_residual(self, z_bc: Tensor, g: Tensor) -> Tensor:
        """Evaluate the BC residual on boundary points ``z_bc``.

        Args:
            z_bc: ``(N_b, d)`` boundary collocation points.
            g: ``(N_b, out)`` target boundary values.

        Returns:
            ``(N_b, out)`` residual tensor.
        """
        u = self(z_bc)  # type: ignore[operator]
        return self.bc_operator(u, z_bc, g)

    def pinn_loss(
        self,
        z_col: Tensor,
        z_bc: Optional[Tensor] = None,
        g_bc: Optional[Tensor] = None,
        z_data: Optional[Tensor] = None,
        u_data: Optional[Tensor] = None,
        *,
        w_pde: float = 1.0,
        w_bc: float = 1.0,
        w_data: float = 1.0,
    ) -> Tensor:
        """Compute the total weighted PINN loss (Eq. 6 of Cuomo et al. 2022).

        L = w_pde * L_F + w_bc * L_B + w_data * L_data

        Args:
            z_col: ``(N_c, d)`` interior collocation points (``requires_grad=True``).
            z_bc: ``(N_b, d)`` boundary/IC points (optional).
            g_bc: ``(N_b, out)`` boundary/IC target values (optional).
            z_data: ``(N_d, d)`` observed data locations (optional).
            u_data: ``(N_d, out)`` observed solution values (optional).
            w_pde: weight for PDE residual loss.
            w_bc: weight for boundary condition loss.
            w_data: weight for data fit loss.

        Returns:
            Scalar loss tensor.
        """
        r_f = self.pde_residual(z_col)
        loss = w_pde * r_f.pow(2).mean()

        if z_bc is not None and g_bc is not None:
            r_b = self.bc_residual(z_bc, g_bc)
            loss = loss + w_bc * r_b.pow(2).mean()

        if z_data is not None and u_data is not None:
            u_pred = self(z_data)  # type: ignore[operator]
            loss = loss + w_data * F.mse_loss(u_pred, u_data)

        return loss


# ---------------------------------------------------------------------------
# PINN: complete module with configurable backbone
# ---------------------------------------------------------------------------

class PINN(PINNMixin, Module):
    r"""Physics-Informed Neural Network (PINN).

    A fully connected feed-forward network that acts as the surrogate solution
    u_theta(z) for a PDE system, trained by minimising the weighted combination
    of PDE residual, boundary-condition, and (optionally) observed-data losses:

    .. math::

        \mathcal{L}(\theta) = \omega_\mathcal{F}\mathcal{L}_\mathcal{F}(\theta)
            + \omega_\mathcal{B}\mathcal{L}_\mathcal{B}(\theta)
            + \omega_d \mathcal{L}_{data}(\theta)

    where each term is the mean-squared error of the corresponding residual
    evaluated at a set of collocation points sampled in the problem domain.

    This implementation follows the **vanilla PINN** of Raissi et al. (2019)
    and the architectural conventions reviewed in Cuomo et al. (2022):

    * ``tanh`` activation (smooth, non-zero second derivatives -- essential for
      second-order PDEs like Poisson, heat, Navier--Stokes).
    * Fully connected (FF-NN / MLP) backbone with configurable depth/width.
    * Soft boundary-condition enforcement via :math:`\mathcal{L}_\mathcal{B}`.
    * Automatic differentiation through :func:`pinn_grad` to build PDE residuals.

    Args:
        in_features: dimensionality of the spatial/temporal input ``z``.
        out_features: dimensionality of the solution ``u``.
        hidden_features: number of neurons per hidden layer.
        num_layers: total number of layers including input projection and output
                    projection (must be >= 2).  The number of hidden layers is
                    ``num_layers - 2``.
        activation: activation module applied after every hidden layer
                    (default: ``nn.Tanh``).  The final output layer has no
                    activation (linear output).
        dropout: dropout probability applied after each hidden activation
                 (default: 0.0, i.e. no dropout).
        layer_norm: if ``True``, apply :class:`LayerNorm` before each hidden
                    activation (default: ``False``).

    Shape:
        - Input: :math:`(*, \text{in\_features})`
        - Output: :math:`(*, \text{out\_features})`

    Examples::

        # Solve 1-D Poisson: -u_xx = f(x) on [0,1] with u(0)=u(1)=0
        import torch
        import torch.nn as nn

        class PoissonPINN(nn.PINN):
            def pde_operator(self, u, z):
                u_xx = pinn_grad(u, z, order=2)
                f = torch.sin(torch.pi * z)   # forcing term
                return -u_xx - f

        model = PoissonPINN(in_features=1, out_features=1,
                            hidden_features=32, num_layers=5)

        z_col  = pinn_collocation(1000, [(0.0, 1.0)])
        z_bc   = torch.tensor([[0.0], [1.0]])
        g_bc   = torch.zeros(2, 1)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2000):
            optimizer.zero_grad()
            loss = model.pinn_loss(z_col, z_bc, g_bc)
            loss.backward()
            optimizer.step()

    Reference:
        Cuomo et al., "Scientific Machine Learning Through Physics-Informed
        Neural Networks: Where we are and What's Next,"
        J. Sci. Comput. 92:88, 2022. https://doi.org/10.1007/s10915-022-01939-z

        Raissi et al., "Physics-informed neural networks: A deep learning
        framework for solving forward and inverse problems involving nonlinear
        PDEs," J. Comput. Phys. 378, 686-707, 2019.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 64,
        num_layers: int = 5,
        activation: Optional[Module] = None,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        if activation is None:
            activation = Tanh()

        layers: list[Module] = []

        # Input projection
        layers.append(Linear(in_features, hidden_features))
        for _ in range(num_layers - 2):
            if layer_norm:
                layers.append(LayerNorm(hidden_features))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(Dropout(p=dropout))
            layers.append(Linear(hidden_features, hidden_features))

        layers.append(activation)
        # Output projection (no activation -- linear readout)
        layers.append(Linear(hidden_features, out_features))

        self.net = Sequential(*layers)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.num_layers = num_layers

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)
