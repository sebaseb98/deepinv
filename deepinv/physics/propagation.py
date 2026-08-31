from __future__ import annotations
import torch
from torch import Tensor
from torch.nn.functional import pad
import numpy as np

from deepinv.physics.forward import Physics

class AngularSpectrumPropagation(Physics):
    def __init__(self, device, **kwargs):
        super().__init__(device=device, **kwargs)


class FresnelPropagation(AngularSpectrumPropagation):
    r"""
    Fresnel propagation operator.

    used to propagate a complex-valued wave field in the near-field regime
    """
    def __init__(
            self, 
            img_size: tuple[int, int],  # H,W
            fresnel_number: float, 
            device: torch.device | str = "cpu",
            **kwargs,
        ):
        super().__init__(device=device, **kwargs)
        self.img_size = img_size
        self.fresnel_number = fresnel_number
        self.device = device
        xi, eta = torch.meshgrid(
                    torch.fft.fftfreq(self.data_shape, device=self.device),
                    torch.fft.fftfreq(self.data_shape, device=self.device),
                    indexing="ij",
                )
        self.kernel_func = torch.exp(
            (-1j * np.pi) / self.fresnel_number * (xi * xi + eta * eta)
        ).to(self.device)

    def A(self, x: Tensor) -> Tensor:
        r"""
        Fresnel propagation operator.
        :param torch.Tensor x: input image (complex-valued)
        :return: (:class:`torch.Tensor`) output measurements
        """
        complex_field = torch.exp(1j*x.real - x.imag)
        propagated = torch.fft.ifft2(
            torch.fft.fft2(complex_field.reshape(-1, *self.img_size))
            * self.kernel_func
        )
        return propagated

    def A_vjp(self, phi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Vector-Jacobian Product (VJP) required for non-linear optimization.

        Computes J_A(phi)^T * v.
        """
        phi = phi.detach().requires_grad_(True)
        with torch.enable_grad():
            y = self.A(phi)
            vjp = torch.autograd.grad(
                outputs=y,
                inputs=phi,
                grad_outputs=v.to(y.device),
                retain_graph=False,
                create_graph=False,
            )[0]
        return vjp
