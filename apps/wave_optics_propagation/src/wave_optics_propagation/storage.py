import json
import os
from datetime import datetime

import numpy as np
from qiskit import qpy
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

PARENT_FOLDER_NAME = ".result"
RESULTS_FILENAME = "results.npz"
INITIAL_PARAMETERS_FILENAME = "initial_parameters.json"
CIRCUIT_FILENAME = "circuit.qpy"


def save_circuit(
    object: QuantumCircuit,
    time_object: datetime,
    wrapper_folder: str | None = None,
):
    timestamp = time_object.strftime("%Y-%m-%d_%H-%M-%S")

    folder_path = (
        os.path.join(PARENT_FOLDER_NAME, timestamp)
        if wrapper_folder is None
        else os.path.join(PARENT_FOLDER_NAME, wrapper_folder, timestamp)
    )
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, CIRCUIT_FILENAME)
    with open(file_path, "wb") as f:
        qpy.dump(object, f)


# def save_results(
#     object: Result,
#     filename: str,
#     time_object: datetime,
#     wrapper_folder: str | None = None,
# ):
#     timestamp = time_object.strftime("%Y-%m-%d_%H-%M-%S")

#     folder_path = (
#         os.path.join(PARENT_FOLDER_NAME, timestamp)
#         if wrapper_folder is None
#         else os.path.join(PARENT_FOLDER_NAME, wrapper_folder, timestamp)
#     )
#     os.makedirs(folder_path, exist_ok=True)

#     file_path = os.path.join(folder_path, f"{filename}.json")
#     with open(file_path, "w") as f:
#         json.dump(object, f)


def save_numpy_results(
    object: dict[str, np.ndarray],
    time_object: datetime,
    wrapper_folder: str | None = None,
):
    timestamp = time_object.strftime("%Y-%m-%d_%H-%M-%S")

    folder_path = os.path.join(
        PARENT_FOLDER_NAME, wrapper_folder if wrapper_folder else timestamp
    )
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, RESULTS_FILENAME)
    np.savez(file_path, **object, allow_pickle=False)


def load_numpy_results(folder_path: str) -> dict[str, np.ndarray]:
    file_path = os.path.join(folder_path, RESULTS_FILENAME)
    data = np.load(file_path, allow_pickle=False)
    return {k: v for k, v in data.items()}


def save_initial_parameters(
    parameters: dict,
    time_object: datetime,
    wrapper_folder: str | None = None,
):
    timestamp = time_object.strftime("%Y-%m-%d_%H-%M-%S")

    folder_path = os.path.join(
        PARENT_FOLDER_NAME, wrapper_folder if wrapper_folder else timestamp
    )
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, INITIAL_PARAMETERS_FILENAME)
    with open(file_path, "w") as f:
        json.dump(parameters, f)


def load_initial_parameters(folder_path: str) -> dict:
    file_path = os.path.join(folder_path, INITIAL_PARAMETERS_FILENAME)
    with open(file_path) as f:
        parameters = json.load(f)
    return parameters
