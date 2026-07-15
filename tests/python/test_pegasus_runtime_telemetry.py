import copy
import sys
import tempfile
import unittest
from pathlib import Path

from gpu_suite.schema import validate_raw_result
from gpu_suite.strict_json import dump_bytes, dumps, load

from pegasus_support import CPU_ENVIRONMENT, write_campaign_inputs
from support import ROOT, raw_success


PEGASUS_DIRECTORY = ROOT / "jobs" / "pegasus"
sys.path.insert(0, str(PEGASUS_DIRECTORY))

from collect_runtime_environment import (  # noqa: E402
    RuntimeEnvironmentError,
    collect_runtime_environment,
)
from telemetry import (  # noqa: E402
    TelemetryError,
    correlate_trials,
    parse_dmon_text,
    write_telemetry_artifacts,
)


def successful_cuda_runtime_probe():
    return {
        "cuda_driver_api_version": "13.0.0",
        "cuda_runtime_version": "13.0.96",
        "diagnostic": None,
        "loaded_library": "/fixture/cuda/lib64/libcudart.so",
        "query_status": "success",
    }


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_complete_document_and_stable_normalized_ldd(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            module_list = directory / "module-list.txt"
            module_list.write_text("runtime/1\nopenmpi/test\n", encoding="utf-8")
            state = {"ldd": 0}

            def which(name):
                return "/fake/" + name

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    state["ldd"] += 1
                    address = "0x0000{0}".format(state["ldd"])
                    return 0, "libfixture.so => /lib/libfixture.so ({0})\n".format(address)
                if name == "pkg-config":
                    return 0, "1.2.3\n"
                if name == "nvidia-smi":
                    return 0, "fixture-value\n"
                return 0, name + " fixture version\n"

            environment = dict(CPU_ENVIRONMENT)
            environment.update({
                "PATH": "/fake/bin",
                "LD_LIBRARY_PATH": "/fake/lib",
                "NVHPC_CUDA_HOME": str(directory / "cuda"),
            })
            first = collect_runtime_environment(
                inputs["manifest"], [inputs["metadata"]], module_list,
                str(directory / "cuda"), "13.0.88",
                str(directory / "cuda"), environment, execute, which,
                require_ldd=True, require_gpu_tools=True,
                cuda_runtime_probe=successful_cuda_runtime_probe,
            )
            second = collect_runtime_environment(
                inputs["manifest"], [inputs["metadata"]], module_list,
                str(directory / "cuda"), "13.0.88",
                str(directory / "cuda"), environment, execute, which,
                require_ldd=True, require_gpu_tools=True,
                cuda_runtime_probe=successful_cuda_runtime_probe,
            )
            self.assertEqual(dump_bytes(first), dump_bytes(second))
            self.assertEqual(first["runtime_environment_schema_version"], 1)
            self.assertEqual(first["cpu_runtime_environment"], CPU_ENVIRONMENT)
            self.assertEqual(
                first["cpu_thread_semantics"],
                {"requested_threads": 48,
                 "serial_cpu_baseline_effective_threads": 1},
            )
            self.assertEqual(first["resolved_shared_library_paths"],
                             ["/lib/libfixture.so"])
            self.assertEqual(first["binaries"][0]["ldd"]["output"],
                             ["libfixture.so => /lib/libfixture.so"])
            self.assertEqual(first["cuda"]["configured_toolkit_version"],
                             "13.0.88")
            self.assertEqual(first["cuda"]["runtime_version"], "13.0.96")
            self.assertEqual(first["cuda"]["runtime_version_source"],
                             "cudaRuntimeGetVersion")
            self.assertEqual(first["nvhpc"]["selected_cuda_toolkit"],
                             str(directory / "cuda"))
            self.assertTrue(all(
                item["status"] == "success"
                for item in first["cpu_library_versions"].values()
            ))

    def test_missing_cpu_runtime_variable_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            module_list = directory / "module-list.txt"
            module_list.write_text("runtime/1\n", encoding="utf-8")
            environment = dict(CPU_ENVIRONMENT)
            del environment["OMP_PLACES"]
            with self.assertRaises(RuntimeEnvironmentError):
                collect_runtime_environment(
                    inputs["manifest"], [inputs["metadata"]], module_list,
                    str(directory / "cuda"), "13.0.88",
                    str(directory / "cuda"),
                    environment,
                )

    def test_runtime_compilers_are_optional_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            module_list = directory / "module-list.txt"
            module_list.write_text("runtime/1\n", encoding="utf-8")

            def which(name):
                if name in {"nvcc", "nvc", "nvc++"}:
                    return None
                return "/fake/" + name

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    return 0, "libfixture.so => /lib/libfixture.so (0x1)\n"
                if name == "nvidia-smi":
                    return 0, "fixture-gpu\n"
                if name == "pkg-config":
                    return 1, ""
                return 0, "fixture\n"

            environment = dict(CPU_ENVIRONMENT)
            environment["NVHPC_CUDA_HOME"] = str(directory / "cuda")
            document = collect_runtime_environment(
                inputs["manifest"], [inputs["metadata"]], module_list,
                str(directory / "cuda"), "13.0.88",
                str(directory / "cuda"), environment, execute, which,
                require_ldd=True, require_gpu_tools=True,
                cuda_runtime_probe=successful_cuda_runtime_probe,
            )
            self.assertEqual(document["cuda"]["nvcc"]["status"],
                             "unavailable")
            self.assertEqual(document["nvhpc"]["compiler_nvc"]["status"],
                             "unavailable")
            with self.assertRaises(RuntimeEnvironmentError):
                collect_runtime_environment(
                    inputs["manifest"], [inputs["metadata"]], module_list,
                    str(directory / "cuda"), "13.0.88",
                    str(directory / "cuda"), environment, execute, which,
                    require_ldd=True, require_gpu_tools=True,
                    require_runtime_compilers=True,
                    cuda_runtime_probe=successful_cuda_runtime_probe,
                )


class TelemetryTests(unittest.TestCase):
    RAW = (
        "# time,gpu,pwr,gtemp,mystery\n"
        "23:59:59,0,300,70,alpha\n"
        "00:00:00,0,301,71,beta\n"
        "00:00:01,0,302,72,gamma\n"
    )

    def test_timezone_midnight_rollover_and_unknown_columns(self):
        header, samples, rollovers = parse_dmon_text(
            self.RAW, "2026-07-14", "+09:00"
        )
        self.assertEqual(header[-1], "mystery")
        self.assertEqual(rollovers, 1)
        self.assertEqual(samples[0]["timestamp_utc"],
                         "2026-07-14T14:59:59.000Z")
        self.assertEqual(samples[1]["timestamp_utc"],
                         "2026-07-14T15:00:00.000Z")
        self.assertEqual(samples[2]["columns"]["mystery"], "gamma")

    def test_non_midnight_clock_reversal_is_rejected(self):
        text = "# time gpu pwr\n10:00:00 0 1\n09:59:59 0 2\n"
        with self.assertRaises(TelemetryError):
            parse_dmon_text(text, "2026-07-14", "+09:00")

    def test_interval_correlation_and_metadata_fields(self):
        header, samples, _ = parse_dmon_text(self.RAW, "2026-07-14", "+09:00")
        record = raw_success()
        record["measurement_start_timestamp"] = "2026-07-14T14:59:59.000Z"
        record["measurement_end_timestamp"] = "2026-07-14T15:00:01.000Z"
        record["elapsed_total_sec"] = 1.0
        record["elapsed_sec"] = 0.5
        record = validate_raw_result(record)
        correlation = correlate_trials([record], samples)[0]
        self.assertEqual(correlation["sample_indices"], [0, 1, 2])
        self.assertEqual(correlation["sample_count"], 3)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw_path = directory / "dmon.txt"
            raw_path.write_text(self.RAW, encoding="utf-8")
            raw_results = directory / "raw.jsonl"
            raw_results.write_text(dumps(record) + "\n", encoding="utf-8")
            metadata = write_telemetry_artifacts(
                raw_path, directory / "metadata.json", directory / "samples.json",
                directory / "correlation.json", raw_results,
                "2026-07-14T14:59:58.000Z",
                "2026-07-14T15:00:02.000Z", 1.0, "JST", "+09:00",
                "2026-07-14", "success", "csv", CPU_ENVIRONMENT,
            )
            self.assertEqual(metadata["timezone"], "JST")
            self.assertEqual(metadata["utc_offset"], "+09:00")
            self.assertEqual(metadata["midnight_rollover_count"], 1)
            self.assertEqual(metadata["cpu_runtime_environment"], CPU_ENVIRONMENT)
            saved = load(directory / "correlation.json")
            self.assertEqual(saved["correlations"][0]["sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
