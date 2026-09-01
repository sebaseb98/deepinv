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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import deepinv as dinv

torch.manual_seed(42)
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
# Getting the data

DATA_PATH = None
if DATA_PATH is None:  # Generate synthetic data
    width, height = 128, 128
    img_size = (1, height, width)
    axis_y = torch.linspace(-1, 1, height, device=device)
    axis_x = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")

    disk = ((grid_x + 0.18).square() + (grid_y + 0.08).square() < 0.32**2).float()
    small_disk = ((grid_x - 0.30).square() + (grid_y - 0.22).square() < 0.14**2).float()
    smooth_feature = torch.exp(
        -((grid_x - 0.20).square() + (grid_y + 0.28).square()) / 0.08
    )

    phase_true = (0.9 * disk + 0.55 * small_disk + 0.35 * smooth_feature)[None, None]
    absorption_true = (0.10 * disk + 0.04 * small_disk)[None, None]
    transmission_true = torch.exp(-absorption_true - 1j * phase_true)


else:  # Load measured data
    measurements = np.load(DATA_PATH)
    img_size = measurements.shape
    measurements = torch.from_numpy(measurements).float().to(device)


# %%
# Defining the Operator
B = MultiDistanceFresnel(
    img_size=img_size,
    distances=distances,
    wavelength=wavelength,
    pixel_size=pixel_size,
    device=device,
)
transmission_phase_retrieval = dinv.physics.PhaseRetrieval(B=B)

if DATA_PATH is None:
    with torch.no_grad():
        measurements = transmission_phase_retrieval.A(transmission_true)
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


# %%
# Plot Multi-distance intensity measurements
# -------------------------------------
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
    phase = F.softplus(parameters[:, 0:1])
    absorption = F.softplus(parameters[:, 1:2])
    return torch.exp(-absorption - 1j * phase)


transmission_model = dinv.physics.Physics(A=parameters_to_transmission)
physics = dinv.physics.compose(transmission_model, transmission_phase_retrieval)

# %%
dinv.utils.plot(
    [
        physics(torch.stack([phase_true, absorption_true], dim=2).squeeze(0)),
        measurements,
    ],
    titles=["Simulated measurements", "measurements"],
    cmap="gray",
)

# %%
# Reconstruction
# --------------
#
# An amplitude-domain data term is less strongly weighted toward bright pixels
# than a direct intensity MSE. Total variation supplies a modest spatial prior.
# The mean phase is removed after every iteration because intensity measurements
# cannot determine a global phase offset.

parameters = torch.zeros(1, 2, height, width, device=device)
parameters = torch.nn.Parameter(parameters)

optimizer = torch.optim.Adam([parameters], lr=0.05)
L1prior = dinv.optim.prior.L1Prior()
n_iter = 1000
loss_history = []
for iteration in range(n_iter):
    optimizer.zero_grad()
    prediction = physics.A(parameters)

    amplitude_residual = torch.sqrt(prediction + 1e-16) - torch.sqrt(
        measurements + 1e-16
    )
    data_loss = amplitude_residual.square().mean()

    phase = parameters[:, 0:1]
    absorption = parameters[:, 1:2]
    loss = data_loss + 2e-2 * L1prior(phase) + 5e-1 * L1prior(absorption)
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

if DATA_PATH is None:
    phase_reference = phase_true - phase_true.mean(dim=(-2, -1), keepdim=True)
    fig, axes = plt.subplots(3, 2, figsize=(8, 7))
    images = [
        phase_reference,
        phase_estimate,
        absorption_true,
        absorption_estimate,
        measurements[:, 0],
        prediction[:, 0].detach(),
    ]
    titles = [
        "True phase",
        "Estimated phase",
        "True absorption",
        "Estimated absorption",
        "Measurement z0",
        "Simulated Measurement z0",
    ]
else:
    fig, axes = plt.subplots(2, 2, figsize=(8, 3))
    images = [
        phase_estimate,
        absorption_estimate,
        measurements[:, 0],
        prediction[:, 0].detach(),
    ]
    titles = [
        "Estimated phase",
        "Estimated absorption",
        "Measurement z0",
        "Simulated Measurement z0",
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
