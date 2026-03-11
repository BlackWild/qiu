# %%

import argparse

from wave_optics_propagation_enhanced.experiment import (
    Experiment,
)
from wave_optics_propagation_enhanced.parameters import (
    ExperimentParameters,
)

# %% Read arguments


parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-delta", type=float, help="Maximum delta value for phase slicing."
)
parser.add_argument(
    "--direct-propagator",
    action="store_true",  # meaning that it is False if not provided
    help="Use direct propagator if set.",
)
parser.add_argument(
    "--reverse-order",
    action="store_true",
    help="Use reverse order for lens slices.",
)
parser.add_argument(
    "--fresnel-approximation",
    action="store_true",
    help="Use Fresnel approximation if set.",
)
args = parser.parse_args()
if args.max_delta is None:
    raise ValueError("Please provide --max-delta argument.")
if args.direct_propagator is None:
    raise ValueError("Please provide --direct-propagator argument.")
if args.reverse_order is None:
    raise ValueError("Please provide --reverse-order argument.")
if args.fresnel_approximation is None:
    raise ValueError("Please provide --fresnel-approximation argument.")

# %% Define experiment parameters

# Scale all lengths, except for the wavelength
# length_scaling = 1.493
length_scaling = 1e-3

experiment_params = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=25e-3 * length_scaling,
    focal_length=200e-3 * length_scaling,
    refractive_index=1.25,
    propagation_after_lens=1.5 * 200e-3 * length_scaling,
    transverse_length=100e-3 * length_scaling,
    num_of_steps_after_lens=300,
    lens_slices=10000,
    num_qubits=7,
    max_delta=args.max_delta,
    lens_reverse_order=args.reverse_order,
    fresnel_approximation=args.fresnel_approximation,
    scale_down_phases=True,
    direct_propagator=args.direct_propagator,
)

print(experiment_params)

# %% Experiment simulation

experiment = Experiment(parameters=experiment_params)

# assert experiment.is_valid(), "Experiment parameters are not valid."

experiment.run(save_results=True)
