import json
import os
from datetime import datetime

from qiskit import qpy
from qiskit.circuit import QuantumCircuit
from qiskit.result import Result

PARENT_FOLDER_NAME = ".result"


def save_circuit(
    object: QuantumCircuit,
    filename: str,
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

    file_path = os.path.join(folder_path, f"{filename}.qpy")
    with open(file_path, "wb") as f:
        qpy.dump(object, f)


def save_results(
    object: Result,
    filename: str,
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

    file_path = os.path.join(folder_path, f"{filename}.json")
    with open(file_path, "w") as f:
        json.dump(object, f)
