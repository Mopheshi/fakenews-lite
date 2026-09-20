import platform

from fakenews.environment import (
    CpuDescription,
    CpuProbe,
    FallbackCpuProbe,
    LinuxCpuProbe,
    MacOsCpuProbe,
    WindowsCpuProbe,
    as_rows,
    describe_environment,
    detect_cpu,
    summarise_hardware,
)


def test_each_probe_claims_only_its_own_platform():
    assert WindowsCpuProbe.supports("Windows")
    assert not WindowsCpuProbe.supports("Linux")
    assert LinuxCpuProbe.supports("Linux")
    assert not LinuxCpuProbe.supports("Darwin")
    assert MacOsCpuProbe.supports("Darwin")
    assert not MacOsCpuProbe.supports("Windows")


def test_the_fallback_claims_everything():
    assert FallbackCpuProbe.supports("Windows")
    assert FallbackCpuProbe.supports("Plan9")


def test_an_unknown_platform_still_produces_a_description():
    description = detect_cpu(system="Plan9")

    assert isinstance(description, CpuDescription)
    assert not description.is_empty


def test_the_probe_for_this_machine_names_a_cpu():
    description = detect_cpu()

    assert description.model


def test_an_empty_description_is_recognised_as_empty():
    assert CpuDescription().is_empty
    assert not CpuDescription(model="Some CPU").is_empty
    assert not CpuDescription(clock_mhz=2800.0).is_empty


class SilentProbe(CpuProbe):
    """A probe on a platform that will not answer."""

    @classmethod
    def supports(cls, system):
        return True

    def describe(self):
        return CpuDescription()


def test_detection_falls_through_a_probe_that_answers_nothing(monkeypatch):
    monkeypatch.setattr(
        "fakenews.environment.CPU_PROBES", (SilentProbe, FallbackCpuProbe), raising=True
    )

    description = detect_cpu(system=platform.system())

    assert not description.is_empty


def test_the_summary_never_claims_a_gpu_it_did_not_count():
    line = summarise_hardware({"cpu_model": "Some CPU", "gpus_visible": None})

    assert "not determined" in line
    assert "GPU present" not in line


def test_the_summary_reports_no_gpu_when_none_were_found():
    assert "no GPU" in summarise_hardware({"cpu_model": "Some CPU", "gpus_visible": 0})


def test_the_summary_does_not_repeat_a_clock_already_in_the_model_name():
    line = summarise_hardware(
        {"cpu_model": "Core i7-1165G7 @ 2.80GHz", "cpu_clock_mhz": 2803.0, "gpus_visible": 0}
    )

    assert line.count("GHz") == 1


def test_the_summary_adds_a_clock_the_model_name_omits():
    line = summarise_hardware(
        {"cpu_model": "Some CPU", "cpu_clock_mhz": 3200.0, "gpus_visible": 0}
    )

    assert "3.20 GHz" in line


def test_rows_flatten_nested_records_with_dotted_names():
    rows = as_rows({"a": 1, "b": {"c": 2, "d": None}})

    assert {"Property": "a", "Value": "1"} in rows
    assert {"Property": "b.c", "Value": "2"} in rows
    assert {"Property": "b.d", "Value": "n/a"} in rows


def test_the_environment_record_carries_what_a_reader_needs_to_reproduce():
    record = describe_environment()

    assert record["hardware"]["cpu_model"]
    assert record["runtime"]["os"]
    assert record["runtime"]["packages"]["numpy"] != "not installed"
    # Every thread variable is reported, set or not, so silence is never ambiguous.
    assert record["runtime"]["thread_environment"]["OMP_NUM_THREADS"]
