"""The command line of the lens experiment simulations, shared by their backends.

The defaults are the experiment of the paper: a beam of 25 um FWHM at 1 um wavelength
through a plano-convex lens of 200 um focal length and refractive index 1.25, in a
transverse window of 100 um, sliced into 10000 slices, followed by 300 um of free space
in 300 steps.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from python_wave_optics.parameters import ExperimentParameters

LENGTH_SCALE = 1e-3
"""The scale of the lengths of the experiment, except for the wavelength."""


def argument_parser(
    description: str, default_results_dir: Path
) -> argparse.ArgumentParser:
    """Return the parser of the simulation's command line."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--max-delta",
        type=float,
        required=True,
        help="Maximum phase per cycle of the sample-based phase protocol.",
    )
    parser.add_argument(
        "--direct-propagator",
        action="store_true",
        help="Apply free propagation directly instead of with the phase protocol.",
    )
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Let the beam enter the lens through its plane side.",
    )
    parser.add_argument(
        "--fresnel-approximation",
        action="store_true",
        help="Approximate the spherical lens surface by a paraboloid.",
    )
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--lens-slices", type=int, default=10000)
    parser.add_argument("--steps-after-lens", type=int, default=300)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=default_results_dir,
        help=f"Directory of the run folders (default: {default_results_dir}).",
    )
    return parser


def parameters_from_arguments(arguments: argparse.Namespace) -> ExperimentParameters:
    """Return the parameters of the experiment of the paper, as the arguments vary it."""
    return ExperimentParameters(
        vacuum_wavelength=1e-6,
        beam_FWHM=25e-3 * LENGTH_SCALE,
        focal_length=200e-3 * LENGTH_SCALE,
        refractive_index=1.25,
        propagation_after_lens=1.5 * 200e-3 * LENGTH_SCALE,
        transverse_length=100e-3 * LENGTH_SCALE,
        num_of_steps_after_lens=arguments.steps_after_lens,
        lens_slices=arguments.lens_slices,
        num_qubits=arguments.num_qubits,
        max_delta=arguments.max_delta,
        lens_reverse_order=arguments.reverse_order,
        fresnel_approximation=arguments.fresnel_approximation,
        scale_down_phases=True,
        direct_propagator=arguments.direct_propagator,
    )


def parse_parameters(
    description: str, default_results_dir: Path, argv: Sequence[str] | None = None
) -> tuple[ExperimentParameters, Path]:
    """Parse the command line into the parameters and the results directory."""
    arguments = argument_parser(description, default_results_dir).parse_args(argv)
    return parameters_from_arguments(arguments), arguments.results_dir
