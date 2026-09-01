r"""
Multi-distance Fresnel phase retrieval
======================================

This example reconstructs projected phase and absorption from photon counts
acquired at several object-to-detector distances. The forward model combines
Fresnel propagation, detector blur and calibration, and Poisson shot noise.
"""

# %%
# Imports and acquisition parameters
# ----------------------------------
#
# All lengths use metres. The wavelength corresponds approximately to a 20 keV
# X-ray beam. ``distances[j]`` must describe the same plane as measurement
# ``j``.
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import deepinv as dinv

torch.manual_seed(0)
device = dinv.utils.get_device()

RESULTS_DIR = Path("results") / "fresnel_phase_retrieval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

wavelength = 6.2e-11
pixel_size = 1.0e-6
distances = (0.01, 0.025, 0.05)
detector_psf_sigmas = (0.6, 0.6, 0.6)  # detector pixels


# %%
# Using measured photon counts
# -----------------------------
#
# Set the paths below to raw photon counts and mean dark and flat fields with
# shape ``(n_distances, H, W)``. The detector calibration is
# :math:`g_j=\mathrm{flat}_j-\mathrm{dark}_j` and
# :math:`b_j=\mathrm{dark}_j`. A known object support can be supplied as a
# boolean array of shape ``(H, W)`` to fix phase and absorption to zero in the
# background.
MEASUREMENTS_PATH: Path | None = None
DARK_PATH: Path | None = None
FLAT_PATH: Path | None = None
SUPPORT_MASK_PATH: Path | None = None

using_measured_data = MEASUREMENTS_PATH is not None
measurements = None

if using_measured_data:
    counts = torch.from_numpy(np.load(MEASUREMENTS_PATH)).float().to(device)
    if counts.ndim != 3 or counts.shape[0] != len(distances):
        raise ValueError(
            "Expected measurements with shape (n_distances, H, W), got "
            f"{tuple(counts.shape)}."
        )
    if DARK_PATH is None or FLAT_PATH is None:
        raise ValueError("DARK_PATH and FLAT_PATH are required for raw counts.")

    dark = torch.from_numpy(np.load(DARK_PATH)).float().to(device)
    flat = torch.from_numpy(np.load(FLAT_PATH)).float().to(device)
    if dark.shape != counts.shape or flat.shape != counts.shape:
        raise ValueError("Measurements, dark fields, and flat fields must match.")
    if not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Photon counts must be finite and nonnegative.")

    detector_background = dark
    detector_gain = flat - dark
    if (
        not torch.isfinite(detector_background).all()
        or not torch.isfinite(detector_gain).all()
        or (detector_background < 0).any()
        or (detector_gain <= 0).any()
    ):
        raise ValueError(
            "Dark fields must be nonnegative and flat fields must exceed them."
        )
    measurements = dinv.utils.TensorList(
        [counts[index][None, None] for index in range(len(distances))]
    )
    height, width = counts.shape[-2:]
else:
    height = width = 64
    incident_photons = torch.tensor((2.0e4, 1.8e4, 1.6e4), device=device)
    detector_gain = incident_photons[:, None, None].expand(-1, height, width)
    detector_background = torch.full_like(detector_gain, 20.0)

img_size = (1, height, width)


# %%
# Projected object and constraints
# --------------------------------
#
# We optimize the dimensionless maps
#
# .. math::
#
#     \phi=k\bar\delta, \qquad \mu=k\bar\beta,
#
# where :math:`k=2\pi/\lambda`. The exit-wave transmission for a plane-wave
# probe is :math:`T=\exp(-\mu-\mathrm{i}\phi)`.
axis_y = torch.linspace(-1, 1, height, device=device)
axis_x = torch.linspace(-1, 1, width, device=device)
grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")

phase_true = None
absorption_true = None
true_parameters = None

if using_measured_data:
    if SUPPORT_MASK_PATH is None:
        support = torch.ones(1, 1, height, width, device=device)
        background_is_fixed = False
    else:
        support = torch.from_numpy(np.load(SUPPORT_MASK_PATH)).to(device)
        if support.shape != (height, width):
            raise ValueError(
                f"Expected support shape {(height, width)}, got {tuple(support.shape)}."
            )
        support = support.bool()[None, None]
        background_is_fixed = True
else:
    support = (grid_x.square() + grid_y.square() < 0.75**2)[None, None]
    background_is_fixed = True

    disk = ((grid_x + 0.18).square() + (grid_y + 0.08).square() < 0.32**2).float()
    small_disk = ((grid_x - 0.30).square() + (grid_y - 0.22).square() < 0.14**2).float()
    smooth_feature = torch.exp(
        -((grid_x - 0.20).square() + (grid_y + 0.28).square()) / 0.08
    )

    phase_true = (0.9 * disk + 0.55 * small_disk + 0.35 * smooth_feature)[
        None, None
    ] * support
    absorption_true = (
        0.01 * support + (0.10 * disk + 0.04 * small_disk)[None, None] * support
    )

    absorption_parameter = torch.log(torch.expm1(absorption_true.clamp_min(1e-6)))
    true_parameters = torch.cat((phase_true, absorption_parameter), dim=1)


def parameters_to_material(parameters):
    """Map unconstrained parameters to projected phase and absorption."""
    phase = parameters[:, 0:1]
    absorption = F.softplus(parameters[:, 1:2])

    if background_is_fixed:
        phase = phase * support
        absorption = absorption * support
    else:
        phase = phase - phase.mean(dim=(-2, -1), keepdim=True)

    return phase, absorption


def parameters_to_transmission(parameters, **kwargs):
    phase, absorption = parameters_to_material(parameters)
    return torch.exp(-absorption - 1j * phase)


# %%
# Stacked forward model
# ---------------------
#
# For each distance, DeepInv composes the nonlinear transmission, linear
# Fresnel propagation, intensity detection, detector point-spread function,
# and affine detector calibration:
#
# .. math::
#
#     \mathcal A_j(\phi,\mu)
#     =g_jQ_j\!\left(\left|P_{z_j}[e^{-\mu-i\phi}]\right|^2\right)+b_j.
#
# Calling ``physics.A`` returns expected photon counts, while ``physics`` also
# applies Poisson shot noise.
transmission = dinv.physics.Physics(A=parameters_to_transmission)
plane_physics = []

for index, (distance, psf_sigma) in enumerate(
    zip(distances, detector_psf_sigmas, strict=True)
):
    propagation = dinv.physics.FresnelPropagation(
        img_size=img_size,
        wavelength=wavelength,
        distance=distance,
        pixel_size=pixel_size,
        device=device,
    )
    intensity = dinv.physics.PhaseRetrieval(B=propagation)

    psf = dinv.physics.functional.gaussian_blur(
        sigma=(psf_sigma, psf_sigma), device=device
    )
    detector_blur = dinv.physics.Blur(filter=psf, padding="circular", device=device)

    gain = detector_gain[index][None, None]
    background = detector_background[index][None, None]
    detector_calibration = dinv.physics.Physics(
        A=lambda x, gain=gain, background=background, **kwargs: gain * x + background,
        noise_model=dinv.physics.PoissonNoise(
            gain=1.0, normalize=False, clip_positive=True
        ),
    )
    plane_physics.append(
        dinv.physics.compose(
            transmission, intensity, detector_blur, detector_calibration
        )
    )

physics = dinv.physics.stack(*plane_physics)

if not using_measured_data:
    with torch.no_grad():
        measurements = physics(true_parameters)

dinv.utils.plot(
    list(measurements),
    titles=[f"z = {1e3 * distance:.0f} mm" for distance in distances],
    save_fn=RESULTS_DIR / "measurements.png",
    figsize=(10, 3),
    cmap="gray",
    cbar=True,
    dpi=200,
    close=True,
)


# %%
# Poisson reconstruction
# ----------------------
#
# The phase uses TV regularization. Absorption uses TV and an additional
# :math:`\ell_1` term to reduce phase-absorption cross-talk.
data_fidelity = dinv.optim.StackedPhysicsDataFidelity(
    [dinv.optim.PoissonLikelihood(gain=1.0, denormalize=False) for _ in distances]
)
tv_prior = dinv.optim.TVL1Prior()
l1_prior = dinv.optim.L1Prior()

lambda_phase_tv = 5e-2
lambda_absorption_tv = 1e-1
lambda_absorption_l1 = 5e-2


def material_regularizer(parameters, *args, **kwargs):
    phase, absorption = parameters_to_material(parameters)
    return (
        lambda_phase_tv * tv_prior(phase)
        + lambda_absorption_tv * tv_prior(absorption)
        + lambda_absorption_l1 * l1_prior(absorption)
    )


prior = dinv.optim.Prior(g=material_regularizer)
initial_parameters = torch.zeros(1, 2, height, width, device=device)
initial_parameters[:, 1].fill_(-4.0)

reconstructor = dinv.optim.GD(
    data_fidelity=data_fidelity,
    prior=prior,
    lambda_reg=1.0,
    stepsize=2e-5,
    max_iter=300,
    backtracking=dinv.optim.BacktrackingConfig(eta=0.5, max_iter=10),
)

parameters_estimate, metrics = reconstructor(
    measurements,
    physics,
    init=initial_parameters,
    compute_metrics=True,
)
phase_estimate, absorption_estimate = parameters_to_material(parameters_estimate)


# %%
# Results
# -------
if using_measured_data:
    images = [phase_estimate / torch.pi, absorption_estimate]
    titles = [r"Estimated phase ($\phi/\pi$)", "Estimated absorption"]
else:
    images = [
        phase_true / torch.pi,
        phase_estimate / torch.pi,
        absorption_true,
        absorption_estimate,
    ]
    titles = [
        r"True phase ($\phi/\pi$)",
        r"Estimated phase ($\phi/\pi$)",
        "True absorption",
        "Estimated absorption",
    ]

dinv.utils.plot(
    images,
    titles=titles,
    save_fn=RESULTS_DIR / "reconstruction.png",
    figsize=(4 * len(images), 3.5),
    cmap="magma",
    cbar=True,
    dpi=200,
    close=True,
)

cost = np.asarray(metrics["cost"][0])
dinv.utils.plot_curves(
    {"Objective change": [(cost - cost[0]).tolist()]},
    save_dir=RESULTS_DIR / "objective_history",
)
print(f"Saved figures to {RESULTS_DIR.resolve()}")
