"""Unit tests for simulator.py."""

from typing import Any

import pytest
import qiu_qiskit_aer_encore.simulator as simulator_module
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerError, AerSimulator
from qiu_qiskit_aer_encore.simulator import (
    GPU_OPTIONS,
    AerDevice,
    aer_simulator,
    available_aer_devices,
)


class RecordingSimulator:
    """Stands in for AerSimulator, recording the options it is created with."""

    def __init__(self, **options: Any) -> None:
        """Record the options."""
        self.options = options


@pytest.fixture
def fake_devices(monkeypatch: pytest.MonkeyPatch):
    """Fake the available devices, and record the simulators instead of creating them."""

    def set_devices(*devices: str) -> None:
        monkeypatch.setattr(simulator_module, "available_aer_devices", lambda: devices)
        monkeypatch.setattr(simulator_module, "AerSimulator", RecordingSimulator)

    return set_devices


class TestAerDevice:
    """Test the AerDevice enum."""

    def test_values(self):
        """Test the raw values of the devices."""
        assert AerDevice.list() == ["auto", "cpu", "gpu"]


class TestAvailableAerDevices:
    """Test available_aer_devices."""

    def test_cpu_is_always_available(self):
        """Test that the CPU is always among the devices."""
        assert "CPU" in available_aer_devices()

    def test_is_cached(self):
        """Test that the devices are only detected once."""
        assert available_aer_devices() is available_aer_devices()


class TestAerSimulator:
    """Test aer_simulator."""

    def test_real_cpu_simulator(self):
        """Test that the CPU simulator is a working Aer simulator."""
        simulator = aer_simulator(device="cpu")
        assert isinstance(simulator, AerSimulator)
        assert simulator.options.device == "CPU"  # type: ignore[union-attr]

        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.measure_all()
        counts = simulator.run(transpile(circuit, simulator), shots=10).result()
        assert counts.get_counts() == {"1": 10}

    @pytest.mark.parametrize("device", [AerDevice.AUTO, AerDevice.CPU, "cpu"])
    def test_cpu_without_gpu(self, fake_devices, device):
        """Test that the CPU is used when no GPU is available."""
        fake_devices("CPU")
        simulator = aer_simulator(device)
        assert simulator.options == {"device": "CPU"}  # type: ignore[attr-defined]

    def test_auto_prefers_gpu(self, fake_devices):
        """Test that a GPU is used with its default options when available."""
        fake_devices("CPU", "GPU")
        simulator = aer_simulator()
        assert simulator.options == {"device": "GPU", **GPU_OPTIONS}  # type: ignore[attr-defined]

    def test_cpu_can_be_forced(self, fake_devices):
        """Test that the CPU is used when requested, even if a GPU is available."""
        fake_devices("CPU", "GPU")
        simulator = aer_simulator(AerDevice.CPU)
        assert simulator.options == {"device": "CPU"}  # type: ignore[attr-defined]

    def test_gpu_without_gpu(self, fake_devices):
        """Test that requesting a GPU fails if none is available."""
        fake_devices("CPU")
        with pytest.raises(AerError, match="GPU"):
            aer_simulator(AerDevice.GPU)

    def test_options_are_passed_and_override_gpu_defaults(self, fake_devices):
        """Test that options reach the simulator, overriding the GPU defaults."""
        fake_devices("CPU", "GPU")
        simulator = aer_simulator("gpu", method="statevector", blocking_enable=False)
        assert simulator.options == {  # type: ignore[attr-defined]
            "device": "GPU",
            **GPU_OPTIONS,
            "method": "statevector",
            "blocking_enable": False,
        }

    def test_invalid_device(self):
        """Test that unknown devices are rejected."""
        with pytest.raises(ValueError):
            aer_simulator("tpu")
