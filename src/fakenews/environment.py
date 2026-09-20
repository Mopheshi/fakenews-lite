"""What machine produced these numbers, and how was it configured?

A training time and a latency figure mean nothing without the machine they were
measured on. ``platform.processor()`` is not enough: on Windows it returns a
family string like ``Intel64 Family 6 Model 140``, which nobody can map to a
purchasable part. So each operating system gets a probe that knows where the
marketing name actually lives, and the probes sit behind one interface so the
callers never branch on ``sys.platform``.

The threading section matters as much as the chip. Numerical work here runs
through OpenBLAS, OpenMP and TensorFlow's own pools, each with its own thread
count, each overridable by an environment variable that may or may not be set.
A latency figure measured with eight BLAS threads available is a different
measurement from one measured with a single thread, and the report should say
which one it is rather than leaving a reader to guess.

Every probe degrades to ``None`` rather than raising. A benchmark that dies
because it could not read a clock speed would be a bad trade.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Thread-count variables worth reporting. Any of these silently changes what a
#: benchmark measures, so an unset one is itself a finding and is recorded.
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)

#: Packages whose version can move a measured number.
REPORTED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "tensorflow",
    "xgboost",
    "nltk",
)

#: Seconds to wait on an external tool before giving up on it.
PROBE_TIMEOUT = 5.0


@dataclass(frozen=True)
class CpuDescription:
    """The part number and clock, as far as the platform will say."""

    model: str | None = None
    clock_mhz: float | None = None
    #: True when the clock is the rated figure, False when it is an
    #: instantaneous reading that will move with load and power state.
    clock_is_rated: bool = True

    @property
    def is_empty(self) -> bool:
        return self.model is None and self.clock_mhz is None


class CpuProbe(ABC):
    """Reads the CPU model name from one operating system."""

    @classmethod
    @abstractmethod
    def supports(cls, system: str) -> bool:
        """Whether this probe applies to ``platform.system()``'s answer."""

    @abstractmethod
    def describe(self) -> CpuDescription:
        """Best effort; never raises."""


class WindowsCpuProbe(CpuProbe):
    """Reads the registry.

    Preferred over WMI or ``wmic``: it is a local key read rather than a service
    call, ``wmic`` is deprecated on Windows 11, and starting PowerShell to ask
    the same question costs a second.
    """

    REGISTRY_PATH = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"

    @classmethod
    def supports(cls, system: str) -> bool:
        return system == "Windows"

    def describe(self) -> CpuDescription:
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, self.REGISTRY_PATH) as key:
                model = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
                megahertz = float(winreg.QueryValueEx(key, "~MHz")[0])
        except Exception:
            logger.debug("Windows CPU probe failed", exc_info=True)
            return CpuDescription()

        # ~MHz is the rated frequency the firmware reports, not a live reading.
        return CpuDescription(model=model or None, clock_mhz=megahertz, clock_is_rated=True)


class LinuxCpuProbe(CpuProbe):
    """Reads ``/proc/cpuinfo``, preferring ``lscpu`` for the rated clock.

    ``cpu MHz`` in ``/proc/cpuinfo`` is whatever the core happened to be running
    at when the file was read, which on a laptop is close to meaningless.
    ``lscpu`` reports the maximum, which is the number a reader can look up.
    """

    @classmethod
    def supports(cls, system: str) -> bool:
        return system == "Linux"

    def describe(self) -> CpuDescription:
        model = self._model_from_cpuinfo()
        clock = self._rated_clock_from_lscpu()

        if clock is None:
            return CpuDescription(
                model=model, clock_mhz=self._current_clock_from_cpuinfo(), clock_is_rated=False
            )
        return CpuDescription(model=model, clock_mhz=clock, clock_is_rated=True)

    def _read_cpuinfo(self) -> str:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            logger.debug("Could not read /proc/cpuinfo", exc_info=True)
            return ""

    def _model_from_cpuinfo(self) -> str | None:
        # "model name" on x86, "Model" or "Hardware" on many ARM boards.
        for field in ("model name", "Model", "Hardware"):
            match = re.search(rf"^{field}\s*:\s*(.+)$", self._read_cpuinfo(), flags=re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None

    def _current_clock_from_cpuinfo(self) -> float | None:
        match = re.search(r"^cpu MHz\s*:\s*([\d.]+)$", self._read_cpuinfo(), flags=re.MULTILINE)
        return float(match.group(1)) if match else None

    def _rated_clock_from_lscpu(self) -> float | None:
        output = _run(["lscpu"])
        if not output:
            return None
        match = re.search(r"^CPU max MHz:\s*([\d.]+)$", output, flags=re.MULTILINE)
        return float(match.group(1)) if match else None


class MacOsCpuProbe(CpuProbe):
    """Reads ``sysctl``.

    Apple silicon does not publish a frequency through ``hw.cpufrequency``, so a
    missing clock here is expected rather than a failure.
    """

    @classmethod
    def supports(cls, system: str) -> bool:
        return system == "Darwin"

    def describe(self) -> CpuDescription:
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        hertz = _run(["sysctl", "-n", "hw.cpufrequency_max"]) or _run(
            ["sysctl", "-n", "hw.cpufrequency"]
        )

        clock_mhz = None
        if hertz and hertz.strip().isdigit():
            clock_mhz = float(hertz.strip()) / 1e6

        return CpuDescription(
            model=model.strip() if model else None, clock_mhz=clock_mhz, clock_is_rated=True
        )


class FallbackCpuProbe(CpuProbe):
    """Whatever the standard library will say, on a platform with no probe."""

    @classmethod
    def supports(cls, system: str) -> bool:
        return True

    def describe(self) -> CpuDescription:
        return CpuDescription(model=platform.processor() or platform.machine() or None)


#: Order matters only in that the fallback must come last.
CPU_PROBES: tuple[type[CpuProbe], ...] = (
    WindowsCpuProbe,
    LinuxCpuProbe,
    MacOsCpuProbe,
    FallbackCpuProbe,
)


def _run(command: list[str]) -> str | None:
    """Run an external probe, returning None if it is absent or misbehaves."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Probe %s unavailable", command[0], exc_info=True)
        return None

    return completed.stdout if completed.returncode == 0 else None


def detect_cpu(system: str | None = None) -> CpuDescription:
    """Describe the CPU using the first probe that claims this platform.

    Falls through to the next probe when one returns nothing, so a locked-down
    registry or a container without ``/proc`` still yields the standard
    library's answer rather than an empty record.
    """
    system = system or platform.system()

    for probe_class in CPU_PROBES:
        if not probe_class.supports(system):
            continue
        description = probe_class().describe()
        if not description.is_empty:
            return description

    return FallbackCpuProbe().describe()


def describe_hardware() -> dict:
    """CPU, cores, memory and accelerator visibility."""
    cpu = detect_cpu()

    profile: dict = {
        "cpu_model": cpu.model,
        "cpu_clock_mhz": cpu.clock_mhz,
        "cpu_clock_basis": "rated" if cpu.clock_is_rated else "instantaneous reading",
        "cpu_identifier": platform.processor() or platform.machine(),
        "architecture": platform.machine(),
    }

    try:
        import psutil

        profile["physical_cores"] = psutil.cpu_count(logical=False)
        profile["logical_cores"] = psutil.cpu_count(logical=True)
        profile["total_ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        profile["physical_cores"] = None
        profile["logical_cores"] = os.cpu_count()
        profile["total_ram_gb"] = None
        logger.warning("psutil not installed; RAM and physical core count unavailable")

    profile["gpus_visible"] = _visible_gpus()
    return profile


def _visible_gpus() -> int | None:
    """GPU count TensorFlow can see, or None if it cannot be asked.

    Only consulted when TensorFlow is already imported. Importing it here just
    to count GPUs would add seconds to any caller that never needed it.
    """
    if "tensorflow" not in sys.modules:
        return None
    try:
        return len(sys.modules["tensorflow"].config.list_physical_devices("GPU"))
    except Exception:
        logger.debug("Could not enumerate GPUs", exc_info=True)
        return None


def describe_runtime() -> dict:
    """Operating system, interpreter, library versions and threading."""
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": _package_versions(),
        "blas": describe_blas(),
        "thread_environment": {
            name: os.environ.get(name, "unset") for name in THREAD_ENVIRONMENT_VARIABLES
        },
        "tensorflow_threads": _tensorflow_threads(),
    }


def _package_versions() -> dict[str, str]:
    import importlib

    versions = {}
    for name in REPORTED_PACKAGES:
        # Only report what is already imported or importable without side
        # effects worth avoiding; a missing package is recorded as such.
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed"
    return versions


def describe_blas() -> list[dict] | None:
    """Which BLAS/OpenMP runtimes are loaded, and with how many threads.

    ``threadpoolctl`` only sees libraries that have actually been loaded, so
    this is worth calling after the numerical work rather than before it. An
    empty list means nothing had loaded yet, which is reported as such rather
    than as "no BLAS".
    """
    try:
        import threadpoolctl
    except ImportError:
        logger.debug("threadpoolctl not installed; BLAS configuration unavailable")
        return None

    fields = ("user_api", "internal_api", "prefix", "version", "threading_layer", "num_threads")
    return [
        {field: entry.get(field) for field in fields if field in entry}
        for entry in threadpoolctl.threadpool_info()
    ]


def _tensorflow_threads() -> dict | None:
    """TensorFlow's op parallelism, where 0 means "TensorFlow decides"."""
    if "tensorflow" not in sys.modules:
        return None

    try:
        threading = sys.modules["tensorflow"].config.threading
        intra = threading.get_intra_op_parallelism_threads()
        inter = threading.get_inter_op_parallelism_threads()

        record = {"intra_op": intra, "inter_op": inter}
        if 0 in (intra, inter):
            record["note"] = "0 means TensorFlow chooses, normally one thread per logical core"
        return record
    except Exception:
        logger.debug("Could not read TensorFlow threading configuration", exc_info=True)
        return None


def describe_environment() -> dict:
    """The full record: ``{"hardware": ..., "runtime": ...}``.

    Runtime is built first on purpose. Counting GPUs needs TensorFlow imported,
    and the version sweep in the runtime section is what imports it; the other
    order reports the GPU count as unknown on a machine that could have answered.
    """
    runtime = describe_runtime()
    return {"hardware": describe_hardware(), "runtime": runtime}


def summarise_hardware(hardware: dict) -> str:
    """One line naming the machine, for a log or a paper's methods section."""
    model = hardware.get("cpu_model") or hardware.get("cpu_identifier") or "unknown CPU"
    clock = hardware.get("cpu_clock_mhz")
    cores = hardware.get("physical_cores")
    threads = hardware.get("logical_cores")
    ram = hardware.get("total_ram_gb")

    parts = [model]
    # Most vendors already put the clock in the model string; repeating it reads
    # as though two different figures are being quoted.
    if clock and "ghz" not in model.lower():
        parts.append(f"{clock / 1000:.2f} GHz")
    if cores and threads:
        parts.append(f"{cores} cores / {threads} threads")
    if ram:
        parts.append(f"{ram} GB RAM")

    gpus = hardware.get("gpus_visible")
    if gpus == 0:
        parts.append("no GPU")
    elif gpus is None:
        # Never silently upgrade "could not ask" into "has one": this line is
        # written to go into a methods section.
        parts.append("GPU visibility not determined")
    else:
        parts.append(f"{gpus} GPU(s) visible")

    return ", ".join(parts)


def as_rows(record: dict, *, prefix: str = "") -> list[dict[str, str]]:
    """Flatten a nested record into ``Property``/``Value`` rows for a table."""
    rows: list[dict[str, str]] = []
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(as_rows(value, prefix=f"{name}."))
        else:
            rows.append({"Property": name, "Value": "n/a" if value is None else str(value)})
    return rows


__all__ = [
    "CpuDescription",
    "CpuProbe",
    "FallbackCpuProbe",
    "LinuxCpuProbe",
    "MacOsCpuProbe",
    "WindowsCpuProbe",
    "as_rows",
    "describe_blas",
    "describe_environment",
    "describe_hardware",
    "describe_runtime",
    "detect_cpu",
    "summarise_hardware",
]
