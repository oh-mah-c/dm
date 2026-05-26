"""Tests for torch.nn.PINN and helpers.

Reference: Cuomo et al., J. Sci. Comput. 92:88 (2022).
"""

import math
import unittest

import torch
import torch.nn as nn
from torch.nn.modules.pinn import pinn_collocation, pinn_grad
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPinnGrad(TestCase):
    """pinn_grad: automatic derivative via torch.autograd."""

    def test_first_order_linear(self):
        # u(x) = 3x + 1  =>  du/dx = 3
        x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        u = 3.0 * x + 1.0
        grad = pinn_grad(u, x)
        self.assertEqual(grad, torch.full_like(x, 3.0))

    def test_second_order_quadratic(self):
        # u(x) = x^2  =>  d^2u/dx^2 = 2
        x = torch.tensor([[0.5], [1.0], [1.5]], requires_grad=True)
        u = x.pow(2)
        grad2 = pinn_grad(u, x, order=2)
        self.assertEqual(grad2, torch.full_like(x, 2.0))

    def test_multivar_partial(self):
        # u(x, y) = x^2 * y  =>  du/dx = 2xy,  du/dy = x^2
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        u = z[:, 0:1].pow(2) * z[:, 1:2]
        grad = pinn_grad(u, z)
        # du/dx at (1,2) = 4,  at (3,4) = 24
        self.assertAlmostEqual(grad[0, 0].item(), 4.0, places=5)
        self.assertAlmostEqual(grad[1, 0].item(), 24.0, places=5)
        # du/dy at (1,2) = 1,  at (3,4) = 9
        self.assertAlmostEqual(grad[0, 1].item(), 1.0, places=5)
        self.assertAlmostEqual(grad[1, 1].item(), 9.0, places=5)

    def test_order_zero_raises(self):
        x = torch.tensor([1.0], requires_grad=True)
        u = x * 2.0
        with self.assertRaises(ValueError):
            pinn_grad(u, x, order=0)

    def test_unused_var_returns_zeros(self):
        # u does not depend on z2 -> gradient should be zero
        z1 = torch.tensor([[1.0]], requires_grad=True)
        z2 = torch.tensor([[2.0]], requires_grad=True)
        u = z1 * 3.0  # independent of z2
        grad = pinn_grad(u, z2)
        self.assertEqual(grad, torch.zeros_like(z2))

    def test_create_graph_allows_backprop(self):
        # Verify residual loss can be backpropagated
        net = nn.Linear(1, 1)
        x = torch.rand(10, 1, requires_grad=True)
        u = net(x)
        du = pinn_grad(u, x, create_graph=True)
        loss = du.pow(2).mean()
        loss.backward()
        self.assertIsNotNone(net.weight.grad)


class TestPinnCollocation(TestCase):
    """pinn_collocation: uniform sampler for collocation points."""

    def test_shape(self):
        pts = pinn_collocation(50, [(0.0, 1.0), (-1.0, 1.0)])
        self.assertEqual(pts.shape, (50, 2))

    def test_range_1d(self):
        pts = pinn_collocation(1000, [(0.5, 2.5)])
        self.assertTrue(pts.min().item() >= 0.5 - 1e-6)
        self.assertTrue(pts.max().item() <= 2.5 + 1e-6)

    def test_range_2d(self):
        domain = [(-1.0, 1.0), (0.0, 3.0)]
        pts = pinn_collocation(2000, domain)
        self.assertTrue(pts[:, 0].min().item() >= -1.0 - 1e-6)
        self.assertTrue(pts[:, 0].max().item() <= 1.0 + 1e-6)
        self.assertTrue(pts[:, 1].min().item() >= 0.0 - 1e-6)
        self.assertTrue(pts[:, 1].max().item() <= 3.0 + 1e-6)

    def test_requires_grad(self):
        pts = pinn_collocation(10, [(0.0, 1.0)])
        self.assertTrue(pts.requires_grad)

    def test_no_grad(self):
        pts = pinn_collocation(10, [(0.0, 1.0)], requires_grad=False)
        self.assertFalse(pts.requires_grad)

    def test_dtype_device(self):
        pts = pinn_collocation(5, [(0.0, 1.0)], dtype=torch.float64)
        self.assertEqual(pts.dtype, torch.float64)


class TestPINN(TestCase):
    """nn.PINN: backbone construction and forward pass."""

    def test_forward_shape(self):
        model = nn.PINN(in_features=2, out_features=1,
                        hidden_features=16, num_layers=4)
        z = torch.randn(8, 2)
        u = model(z)
        self.assertEqual(u.shape, (8, 1))

    def test_forward_shape_multiout(self):
        model = nn.PINN(in_features=3, out_features=2,
                        hidden_features=8, num_layers=2)
        z = torch.randn(5, 3)
        u = model(z)
        self.assertEqual(u.shape, (5, 2))

    def test_invalid_num_layers(self):
        with self.assertRaises(ValueError):
            nn.PINN(in_features=1, out_features=1, num_layers=1)

    def test_layer_norm(self):
        model = nn.PINN(in_features=1, out_features=1,
                        hidden_features=8, num_layers=3,
                        layer_norm=True)
        z = torch.randn(4, 1)
        u = model(z)
        self.assertEqual(u.shape, (4, 1))

    def test_dropout(self):
        model = nn.PINN(in_features=2, out_features=1,
                        hidden_features=16, num_layers=4,
                        dropout=0.1)
        model.train()
        z = torch.randn(20, 2)
        u = model(z)
        self.assertEqual(u.shape, (20, 1))

    def test_custom_activation(self):
        model = nn.PINN(in_features=1, out_features=1,
                        hidden_features=8, num_layers=3,
                        activation=nn.SiLU())
        z = torch.randn(4, 1)
        u = model(z)
        self.assertEqual(u.shape, (4, 1))

    def test_parameters_exist(self):
        model = nn.PINN(in_features=2, out_features=1,
                        hidden_features=16, num_layers=3)
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_is_nn_module(self):
        model = nn.PINN(in_features=1, out_features=1)
        self.assertIsInstance(model, nn.Module)


class TestPINNMixin(TestCase):
    """PINNMixin: default operators and pinn_loss."""

    def _make_model(self):
        class SimplePINN(nn.modules.pinn.PINNMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Linear(1, 1)

            def forward(self, z):
                return self.net(z)

        return SimplePINN()

    def test_pde_residual_shape(self):
        model = self._make_model()
        z = torch.rand(20, 1, requires_grad=True)
        r = model.pde_residual(z)
        self.assertEqual(r.shape, (20, 1))

    def test_bc_residual_default_dirichlet(self):
        model = self._make_model()
        z_bc = torch.tensor([[0.0], [1.0]])
        g = torch.zeros(2, 1)
        r = model.bc_residual(z_bc, g)
        # default BC: u - g; r = model(z_bc) - 0
        u = model(z_bc)
        self.assertEqual(r, u)

    def test_pinn_loss_pde_only(self):
        model = self._make_model()
        z_col = torch.rand(30, 1, requires_grad=True)
        loss = model.pinn_loss(z_col)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(loss.shape, ())

    def test_pinn_loss_with_bc(self):
        model = self._make_model()
        z_col = torch.rand(30, 1, requires_grad=True)
        z_bc = torch.tensor([[0.0], [1.0]])
        g_bc = torch.zeros(2, 1)
        loss = model.pinn_loss(z_col, z_bc, g_bc)
        self.assertTrue(loss.item() >= 0.0)

    def test_pinn_loss_with_data(self):
        model = self._make_model()
        z_col = torch.rand(30, 1, requires_grad=True)
        z_data = torch.rand(10, 1)
        u_data = torch.rand(10, 1)
        loss = model.pinn_loss(z_col, z_data=z_data, u_data=u_data)
        self.assertTrue(loss.item() >= 0.0)

    def test_pinn_loss_weights(self):
        model = self._make_model()
        z_col = torch.rand(20, 1, requires_grad=True)
        z_bc = torch.tensor([[0.0], [1.0]])
        g_bc = torch.zeros(2, 1)
        loss_1 = model.pinn_loss(z_col, z_bc, g_bc, w_pde=1.0, w_bc=1.0)
        loss_2 = model.pinn_loss(z_col, z_bc, g_bc, w_pde=2.0, w_bc=2.0)
        # doubling weights doubles (approximately) the loss
        self.assertTrue(loss_2.item() > loss_1.item())

    def test_pinn_loss_backward(self):
        model = self._make_model()
        z_col = torch.rand(30, 1, requires_grad=True)
        loss = model.pinn_loss(z_col)
        loss.backward()
        self.assertIsNotNone(model.net.weight.grad)


class TestPINNPoissonSolver(TestCase):
    """Integration test: PINN solves 1-D Poisson -u_xx = f on [0,1], u(0)=u(1)=0.

    Exact solution:  u*(x) = sin(pi*x) / pi^2
    Forcing:         f(x)  = sin(pi*x)
    """

    def test_poisson_convergence(self):
        class PoissonPINN(nn.PINN):
            def pde_operator(self, u, z):
                # -d^2u/dx^2 - sin(pi*x)
                u_xx = pinn_grad(u, z, order=2)
                f = torch.sin(math.pi * z)
                return -u_xx - f

        torch.manual_seed(0)
        model = PoissonPINN(in_features=1, out_features=1,
                            hidden_features=32, num_layers=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        z_bc = torch.tensor([[0.0], [1.0]])
        g_bc = torch.zeros(2, 1)

        for _ in range(3000):
            z_col = pinn_collocation(200, [(0.0, 1.0)])
            optimizer.zero_grad()
            loss = model.pinn_loss(z_col, z_bc, g_bc, w_pde=1.0, w_bc=10.0)
            loss.backward()
            optimizer.step()

        # Evaluate relative L2 error against exact solution
        model.eval()
        with torch.no_grad():
            x_test = torch.linspace(0.0, 1.0, 200).unsqueeze(1)
            u_pred = model(x_test)
            u_exact = torch.sin(math.pi * x_test) / (math.pi ** 2)
            rel_err = (u_pred - u_exact).pow(2).mean().sqrt() / u_exact.pow(2).mean().sqrt()

        # Expect < 5% relative error after 3000 Adam steps
        self.assertLess(rel_err.item(), 0.05)


class TestPINNExportedFromNN(TestCase):
    """Verify PINN, PINNMixin, pinn_grad, pinn_collocation are in torch.nn."""

    def test_pinn_in_nn(self):
        self.assertTrue(hasattr(nn, "PINN"))
        self.assertTrue(issubclass(nn.PINN, nn.Module))

    def test_pinn_mixin_in_nn(self):
        self.assertTrue(hasattr(nn, "PINNMixin"))

    def test_helpers_importable_from_modules(self):
        from torch.nn.modules.pinn import pinn_collocation as pc
        from torch.nn.modules.pinn import pinn_grad as pg
        self.assertTrue(callable(pc))
        self.assertTrue(callable(pg))


if __name__ == "__main__":
    run_tests()
