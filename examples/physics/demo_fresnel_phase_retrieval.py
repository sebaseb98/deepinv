r"""
Multi-distance Fresnel phase retrieval
======================================

This example reconstructs a complex transmission image from intensity
measurements acquired at several object-to-detector distances. It shows both
the tensor convention expected for experimental data and the separation
between

* linear Fresnel propagation of a complex transmission, and
* the nonlinear map from projected phase/absorption to that transmission.

The example generates synthetic measurements by default. The section
`Using measured intensities`_ shows how to replace them with flat-field
corrected measurements from disk.
"""

# %%
# Imports and acquisition parameters
# ----------------------------------
#
# All lengths below use metres. The wavelength corresponds approximately to a
# 20 keV X-ray beam. ``distances[j]`` must describe the same plane as
# ``measurements[j]``.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import deepinv as dinv

torch.manual_seed(0)
device = dinv.utils.get_device()

wavelength = 6.2e-11
pixel_size = 1.0e-6
distances = (0.01, 0.025, 0.05)


# %%
# A linear multi-distance propagation operator
# --------------------------------------------
#
# :class:`deepinv.physics.FresnelPropagation` represents one propagation
# distance. The following lightweight wrapper stacks several propagated fields
# along dimension 1, giving tensors with shape ``(batch, z, channel, H, W)``.
# Its adjoint sums the back-propagated fields over all distances.
class MultiDistanceFresnel(dinv.physics.LinearPhysics):
    r"""Stack Fresnel fields evaluated at several propagation distances."""

    def __init__(
        self,
        img_size,
        distances,
        wavelength,
        pixel_size,
        device="cpu",
    ):
        distances = tuple(float(z) for z in distances)
        if not distances:
            raise ValueError("At least one propagation distance is required.")

        super().__init__(img_size=img_size, device=device)
        self.propagators = torch.nn.ModuleList(
            [
                dinv.physics.FresnelPropagation(
                    img_size=img_size,
                    wavelength=wavelength,
                    distance=z,
                    pixel_size=pixel_size,
                    device=device,
                )
                for z in distances
            ]
        )
        self.register_buffer(
            "distances", torch.tensor(distances, dtype=torch.float32, device=device)
        )

    def A(self, x, **kwargs):
        return torch.stack([propagator.A(x) for propagator in self.propagators], dim=1)

    def A_adjoint(self, y, **kwargs):
        if y.shape[1] != len(self.propagators):
            raise ValueError(
                f"Expected {len(self.propagators)} distance planes, got {y.shape[1]}."
            )
        return sum(
            propagator.A_adjoint(y[:, index])
            for index, propagator in enumerate(self.propagators)
        )

    def A_dagger(self, y, **kwargs):
        # Every Fresnel transfer function has unit magnitude, hence
        # B^* B = n_distances I for the vertically stacked operator B.
        return self.A_adjoint(y) / len(self.propagators)


# %%
# Using measured intensities
# --------------------------
#
# Set these paths to arrays of raw, dark, and flat images with shape
# ``(n_distances, H, W)``. If the first array is already flat-field normalized,
# set ``MEASUREMENTS_ARE_NORMALIZED=True`` and leave the other paths unset.
# Leave ``MEASUREMENTS_PATH`` as ``None`` to run the synthetic example. The
# loading branch infers ``H`` and ``W`` from the data.
#
# A detector mask can be used to exclude dead, saturated, or otherwise invalid
# pixels from the loss below. With magnifying cone-beam geometry, use the
# effective object-plane pixel size and the corresponding effective propagation
# distance rather than the raw detector pitch and sample-detector distance.
MEASUREMENTS_PATH: Path | None = None
DARK_PATH: Path | None = None
FLAT_PATH: Path | None = None
MEASUREMENTS_ARE_NORMALIZED = False

using_measured_data = MEASUREMENTS_PATH is not None
measurements = None

if using_measured_data:
    counts = torch.from_numpy(np.load(MEASUREMENTS_PATH)).float().to(device)
    if counts.ndim != 3 or counts.shape[0] != len(distances):
        raise ValueError(
            "Expected raw measurements with shape (n_distances, H, W), got "
            f"{tuple(counts.shape)}."
        )

    if MEASUREMENTS_ARE_NORMALIZED:
        normalized = counts.clamp_min(0)
    else:
        if DARK_PATH is None or FLAT_PATH is None:
            raise ValueError(
                "DARK_PATH and FLAT_PATH are required for raw measurements."
            )
        dark = torch.from_numpy(np.load(DARK_PATH)).float().to(device)
        flat = torch.from_numpy(np.load(FLAT_PATH)).float().to(device)
        normalized = ((counts - dark) / (flat - dark).clamp_min(1e-6)).clamp_min(0)

    measurements = normalized[None, :, None]
    height, width = counts.shape[-2:]
else:
    height = width = 64

img_size = (1, height, width)


# %%
# Synthetic projected object
# --------------------------
#
# We reconstruct dimensionless projected phase and absorption,
#
# .. math::
#
#     \phi = k\int\delta\,\mathrm{d}z, \qquad
#     \mu = k\int\beta\,\mathrm{d}z,
#
# so that the exit-wave transmission is
# :math:`T=\exp(-\mu-\mathrm{i}\phi)`. Optimizing these dimensionless
# quantities is usually better conditioned than optimizing the very small
# refractive-index decrements directly.
phase_true = None
absorption_true = None
transmission_true = None

if not using_measured_data:
    axis_y = torch.linspace(-1, 1, height, device=device)
    axis_x = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")

    disk = ((grid_x + 0.18).square() + (grid_y + 0.08).square() < 0.32**2).float()
    small_disk = ((grid_x - 0.30).square() + (grid_y - 0.22).square() < 0.14**2).float()
    smooth_feature = torch.exp(
        -((grid_x - 0.20).square() + (grid_y + 0.28).square()) / 0.08
    )

    phase_true = (0.9 * disk - 0.55 * small_disk + 0.35 * smooth_feature)[None, None]
    absorption_true = (0.10 * disk + 0.04 * small_disk)[None, None]
    transmission_true = torch.exp(-absorption_true - 1j * phase_true)


# %%
# Multi-distance intensity measurements
# -------------------------------------
#
# ``B`` is linear in the complex transmission. The existing
# :class:`deepinv.physics.PhaseRetrieval` class then adds the detector
# intensity :math:`|BT|^2`.
B = MultiDistanceFresnel(
    img_size=img_size,
    distances=distances,
    wavelength=wavelength,
    pixel_size=pixel_size,
    device=device,
)
transmission_phase_retrieval = dinv.physics.PhaseRetrieval(B=B)

if not using_measured_data:
    with torch.no_grad():
        clean_intensity = transmission_phase_retrieval.A(transmission_true)
        photons_per_pixel = 2.0e4
        measurements = (
            torch.poisson(photons_per_pixel * clean_intensity) / photons_per_pixel
        )

fig, axes = plt.subplots(1, len(distances), figsize=(10, 3))
for index, (axis, distance) in enumerate(zip(axes, distances, strict=True)):
    image = axis.imshow(measurements[0, index, 0].cpu(), cmap="gray")
    axis.set_title(f"z = {1e3 * distance:.0f} mm")
    axis.axis("off")
    fig.colorbar(image, ax=axis, fraction=0.046)
plt.tight_layout()
plt.show()


# %%
# Nonlinear transmission model
# ----------------------------
#
# The reconstruction variable has two real channels: phase and an unconstrained
# absorption parameter. ``softplus`` makes the physical absorption nonnegative.
# This map is nonlinear, so it is composed *before* the phase-retrieval operator
# and is not included in its linear operator ``B``.
def parameters_to_transmission(parameters):
    phase = parameters[:, 0:1]
    absorption = F.softplus(parameters[:, 1:2])
    return torch.exp(-absorption - 1j * phase)


transmission_model = dinv.physics.Physics(A=parameters_to_transmission)
physics = dinv.physics.compose(transmission_model, transmission_phase_retrieval)


# %%
# Reconstruction
# --------------
#
# An amplitude-domain data term is less strongly weighted toward bright pixels
# than a direct intensity MSE. Total variation supplies a modest spatial prior.
# The mean phase is removed after every iteration because intensity measurements
# cannot determine a global phase offset.
def total_variation(x):
    vertical = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    horizontal = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return vertical + horizontal


parameters = torch.zeros(1, 2, height, width, device=device)
parameters[:, 1].fill_(-5.0)  # softplus(-5) is weak initial absorption
parameters = torch.nn.Parameter(parameters)

optimizer = torch.optim.Adam([parameters], lr=0.05)
valid_pixels = torch.isfinite(measurements) & (measurements >= 0)
measurements = torch.nan_to_num(measurements, nan=0.0, posinf=0.0, neginf=0.0)

n_iter = 500
loss_history = []
for iteration in range(n_iter):
    optimizer.zero_grad()
    prediction = physics.A(parameters)

    amplitude_residual = torch.sqrt(prediction.clamp_min(1e-8)) - torch.sqrt(
        measurements.clamp_min(0) + 1e-8
    )
    data_loss = amplitude_residual[valid_pixels].square().mean()

    phase = parameters[:, 0:1]
    absorption = F.softplus(parameters[:, 1:2])
    regularization = 2e-4 * total_variation(phase) + 5e-4 * total_variation(absorption)
    loss = data_loss + regularization
    loss.backward()
    optimizer.step()

    # Fix the unobservable global phase gauge.
    with torch.no_grad():
        parameters[:, 0:1] -= parameters[:, 0:1].mean(dim=(-2, -1), keepdim=True)

    loss_history.append(loss.detach().cpu())
    if iteration % 50 == 0 or iteration == n_iter - 1:
        print(f"iteration {iteration:3d}: loss={loss.item():.3e}")


# %%
# Results
# -------
phase_estimate = parameters[:, 0:1].detach()
absorption_estimate = F.softplus(parameters[:, 1:2]).detach()

if using_measured_data:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    images = [phase_estimate, absorption_estimate]
    titles = ["Estimated phase", "Estimated absorption"]
else:
    phase_reference = phase_true - phase_true.mean(dim=(-2, -1), keepdim=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    images = [phase_reference, phase_estimate, absorption_true, absorption_estimate]
    titles = [
        "True phase",
        "Estimated phase",
        "True absorption",
        "Estimated absorption",
    ]

for axis, image, title in zip(np.asarray(axes).ravel(), images, titles, strict=True):
    artist = axis.imshow(image[0, 0].cpu(), cmap="magma")
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(artist, ax=axis, fraction=0.046)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5, 3))
plt.semilogy(torch.stack(loss_history))
plt.xlabel("Iteration")
plt.ylabel("Objective")
plt.tight_layout()
plt.show()
