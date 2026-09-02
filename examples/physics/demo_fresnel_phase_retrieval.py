r"""
Multi-distance Fresnel phase retrieval
======================================

This example reconstructs projected phase and absorption from flat-field-
corrected intensities acquired at several object-to-detector distances. We
assume an ideal detector and plane-wave illumination, so Poisson shot noise is
the only detector effect.
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
from scipy.constants import h, c

import deepinv as dinv

torch.manual_seed(0)
device = dinv.utils.get_device()

RESULTS_DIR = Path("results") / "fresnel_phase_retrieval"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

wavelength = h * c / 8e3  # 8 keV
pixel_size = 6.5e-6 / 33.1  # 6.5 um detector pixel size, 33.1x magnification
distances = (156e-3, 158e-3, 166e-3, 187e-3)
photons_per_pixel = 2.0e4
poisson_gain = 1.0 / photons_per_pixel


# %%
# Using measured intensities
# --------------------------
#
# Set ``MEASUREMENTS_PATH`` to the already dark- and flat-field-corrected
# relative intensities with shape ``(n_distances, H, W)`` (the incident field is
# normalized to one). ``photons_per_pixel`` is the corresponding effective
# incident fluence. No calibration fields, detector blur, or additive background
# are modelled.
MEASUREMENTS_PATH: Path | None = Path(
    "/data/dust/user/eberlese/Github/random-PhD-stuff/datasets/holograms_beads_updated.npz"
)

using_measured_data = MEASUREMENTS_PATH is not None
measurements = None

if using_measured_data:
    with np.load(MEASUREMENTS_PATH) as data:
        corrected_intensity = torch.from_numpy(data["holograms"]).float().to(device)
    measurements = dinv.utils.TensorList(
        [measurement[None, None] for measurement in corrected_intensity]
    )
    height, width = corrected_intensity.shape[-2:]
else:
    height = width = 64

img_size = (1, height, width)


# %%
# Projected object and parameterization
# -------------------------------------
#
# We optimize the dimensionless maps
#
# .. math::
#
#     \phi=k\bar\delta, \qquad \mu=k\bar\beta,
#
# where :math:`k=2\pi/\lambda`. For the convention used here,
# :math:`\bar\delta\geq0` gives :math:`\phi\geq0`, while the phase angle of the
# exit wave is :math:`-\phi`. The transmission for a plane-wave probe is
# :math:`T=\exp(-\mu-\mathrm{i}\phi)`.
axis_y = torch.linspace(-1, 1, height, device=device)
axis_x = torch.linspace(-1, 1, width, device=device)
grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")

phase_true = None
absorption_true = None
true_parameters = None

if not using_measured_data:
    phantom_region = (grid_x.square() + grid_y.square() < 0.75**2)[None, None]

    disk = ((grid_x + 0.18).square() + (grid_y + 0.08).square() < 0.32**2).float()
    small_disk = ((grid_x - 0.30).square() + (grid_y - 0.22).square() < 0.14**2).float()
    smooth_feature = torch.exp(
        -((grid_x - 0.20).square() + (grid_y + 0.28).square()) / 0.08
    )

    phase_true = (0.9 * disk + 0.55 * small_disk + 0.35 * smooth_feature)[
        None, None
    ] * phantom_region
    absorption_true = (
        0.01 * phantom_region
        + (0.10 * disk + 0.04 * small_disk)[None, None] * phantom_region
    )

    phase_parameter = torch.log(torch.expm1(phase_true + 1e-6))
    absorption_parameter = torch.log(torch.expm1(absorption_true.clamp_min(1e-6)))
    true_parameters = torch.cat((phase_parameter, absorption_parameter), dim=1)


# Fresnel intensities cannot determine a spatially constant phase offset. We
# choose the representative with minimum phase zero; unlike mean-centering,
# this gauge preserves the physical constraint phi >= 0 without requiring a
# known support mask.
def parameters_to_material(parameters):
    """Map unconstrained parameters to projected phase and absorption."""
    phase = F.softplus(parameters[:, 0:1])
    phase = phase - phase.amin(dim=(-2, -1), keepdim=True)
    absorption = F.softplus(parameters[:, 1:2])

    return phase, absorption


def parameters_to_transmission(parameters, **kwargs):
    phase, absorption = parameters_to_material(parameters)
    return torch.exp(-absorption - 1j * phase)


# %%
# Stacked forward model
# ---------------------
#
# For ideal plane-wave illumination, :math:`p=1`. For each distance, DeepInv
# composes the nonlinear transmission, linear Fresnel propagation, and
# intensity detection:
#
# .. math::
#
#     \mathcal A_j(\phi,\mu)
#     =\left|P_{z_j}[e^{-\mu-i\phi}]\right|^2.
#
# ``PoissonNoise`` is applied directly to each ideal intensity and returns
# normalized counts
# :math:`y_j=\gamma\operatorname{Poisson}(\mathcal A_j/\gamma)`, where
# :math:`\gamma=1/n_{\mathrm{photons}}`. Thus Poisson shot noise is the only
# detector model.
transmission = dinv.physics.Physics(A=parameters_to_transmission)
plane_physics = []

for distance in distances:
    propagation = dinv.physics.FresnelPropagation(
        img_size=img_size,
        wavelength=wavelength,
        distance=distance,
        pixel_size=pixel_size,
        device=device,
    )
    intensity = dinv.physics.PhaseRetrieval(
        B=propagation,
        noise_model=dinv.physics.PoissonNoise(
            gain=poisson_gain,
            normalize=True,
        ),
    )
    plane_physics.append(dinv.physics.compose(transmission, intensity))

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
# Since measurements and predictions are normalized intensities, we minimize
# the Poisson negative log-likelihood per incident photon. This is the raw-count
# likelihood divided by ``photons_per_pixel`` and therefore has the same
# minimizer when the regularization weights are scaled consistently.
#
# The phase uses TV regularization. Absorption uses TV and an additional
# :math:`\ell_1` term to reduce phase-absorption cross-talk.
data_fidelity = dinv.optim.StackedPhysicsDataFidelity(
    [dinv.optim.PoissonLikelihood(gain=1.0, denormalize=False) for _ in distances]
)
tv_prior = dinv.optim.TVL1Prior()
l1_prior = dinv.optim.L1Prior()

lambda_phase_l1 = 0 * 5e-2 / photons_per_pixel
lambda_absorption_tv = 1e-1 / photons_per_pixel
lambda_absorption_l1 = 5e-2 / photons_per_pixel


def material_regularizer(parameters, *args, **kwargs):
    phase, absorption = parameters_to_material(parameters)
    return (
        lambda_phase_l1 * l1_prior(phase)
        + lambda_absorption_tv * tv_prior(absorption)
        + lambda_absorption_l1 * l1_prior(absorption)
    )


prior = dinv.optim.Prior(g=material_regularizer)
initial_parameters = torch.full((1, 2, height, width), -4.0, device=device)

reconstructor = dinv.optim.GD(
    data_fidelity=data_fidelity,
    prior=prior,
    lambda_reg=1.0,
    stepsize=2e-5 * photons_per_pixel,
    max_iter=300,
    backtracking=dinv.optim.BacktrackingConfig(eta=0.5, max_iter=10),
    verbose=True,
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
