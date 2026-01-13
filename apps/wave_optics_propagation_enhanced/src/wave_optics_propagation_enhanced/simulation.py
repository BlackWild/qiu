# %%

from wave_optics_propagation_enhanced.experiment import (
    Experiment,
)
from wave_optics_propagation_enhanced.parameters import (
    ExperimentParameters,
)

# %% Define experiment parameters

experiment_params = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=20e-3 * 1e-3,
    focal_length=200e-3 * 1e-3,
    refractive_index=1.25,
    propagation_after_lens=2 * 200e-3 * 1e-3,
    transverse_length=100e-3 * 1e-3,
    num_of_steps_after_lens=300,
    lens_slices=10000,
    num_qubits=6,
    max_delta=0.01,
    lens_reverse_order=True,
    fresnel_approximation=True,
    scale_down_phases=True,
)


# %% Experiment simulation

experiment = Experiment(parameters=experiment_params)

# assert experiment.is_valid(), "Experiment parameters are not valid."

experiment.run(save_results=True)
