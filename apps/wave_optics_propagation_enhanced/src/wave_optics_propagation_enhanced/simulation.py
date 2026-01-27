# %%

from wave_optics_propagation_enhanced.experiment import (
    Experiment,
)
from wave_optics_propagation_enhanced.parameters import (
    ExperimentParameters,
)

# %% Define experiment parameters

# Scale all lengths, except for the wavelength
# length_scaling = 1.493
length_scaling = 1e-3

experiment_params = ExperimentParameters(
    vacuum_wavelength=1e-6,
    beam_FWHM=23e-3 * length_scaling,
    focal_length=200e-3 * length_scaling,
    refractive_index=1.25,
    propagation_after_lens=1.5 * 200e-3 * length_scaling,
    transverse_length=100e-3 * length_scaling,
    num_of_steps_after_lens=300,
    lens_slices=10000,
    num_qubits=7,
    max_delta=0.1,
    lens_reverse_order=True,
    fresnel_approximation=False,
    scale_down_phases=True,
)


# %% Experiment simulation

experiment = Experiment(parameters=experiment_params)

# assert experiment.is_valid(), "Experiment parameters are not valid."

experiment.run(save_results=True)
