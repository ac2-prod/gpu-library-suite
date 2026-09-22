import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gpu_suite.hashing import deterministic_json_sha256
from gpu_suite.schema import validate_raw_result
from gpu_suite.strict_json import dump_bytes, dumps, load

from pegasus_support import CPU_ENVIRONMENT, write_campaign_inputs
from support import ROOT, raw_success


PEGASUS_DIRECTORY = ROOT / "jobs" / "pegasus"
sys.path.insert(0, str(PEGASUS_DIRECTORY))

from collect_runtime_environment import (  # noqa: E402
    RuntimeEnvironmentError,
    collect_runtime_environment,
    collect_runtime_environment_documents,
    normalize_ldd_output,
    normalize_module_list,
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


def collect_fixture_documents(
    directory, inputs, execute, environment=None,
    module_text="runtime/1\nopenmpi/test\n",
):
    module_list = directory / "module-list-fixture.txt"
    module_list.write_text(module_text, encoding="utf-8")
    selected_environment = dict(CPU_ENVIRONMENT)
    selected_environment.update({
        "PATH": "/fake/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/fake/lib:/usr/lib",
        "NVHPC_CUDA_HOME": str(directory / "cuda"),
    })
    if environment is not None:
        selected_environment.update(environment)
    return collect_runtime_environment_documents(
        inputs["manifest"], [inputs["metadata"]], module_list,
        str(directory / "cuda"), "13.0.88", str(directory / "cuda"),
        selected_environment, execute, lambda name: "/fake/" + name,
        require_ldd=True, require_gpu_tools=True,
        cuda_runtime_probe=successful_cuda_runtime_probe,
    )


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_explicit_module_free_environment_preserves_required_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            environment = dict(CPU_ENVIRONMENT)
            environment.update({"PATH": "/usr/bin", "LD_LIBRARY_PATH": ""})

            def collect(execute, values=environment):
                return collect_runtime_environment_documents(
                    inputs["manifest"], [inputs["metadata"]], None,
                    str(directory / "cuda"), "13.0.88", str(directory / "cuda"),
                    values, execute, lambda name: "/fake/" + name,
                    require_ldd=True, require_gpu_tools=True,
                    cuda_runtime_probe=successful_cuda_runtime_probe,
                )

            document, evidence = collect(lambda arguments: (0, "fixture\n"))
            self.assertEqual(document["module_list"], [])
            self.assertEqual(document["module_system"], {
                "source": "user-specified", "status": "not-used",
            })
            self.assertIsNone(evidence["module_list_output"])
            self.assertIsNone(evidence["module_list_sha256"])
            self.assertEqual(evidence["module_system"], document["module_system"])
            self.assertEqual(
                evidence["runtime_environment_sha256"],
                deterministic_json_sha256(document),
            )
            with self.assertRaisesRegex(RuntimeEnvironmentError, "ldd failed"):
                collect(lambda arguments: (1, "missing dependency tool\n"))
            with self.assertRaisesRegex(RuntimeEnvironmentError, "runtime probe failed"):
                collect(lambda arguments: (
                    (1, "GPU probe failed\n") if "nvidia-smi" in arguments[0]
                    else (0, "fixture\n")
                ))
            with self.assertRaisesRegex(RuntimeEnvironmentError, "missing CPU runtime"):
                collect(lambda arguments: (0, "fixture\n"), {})

    def test_empty_module_output_is_not_a_module_free_declaration(self):
        with self.assertRaisesRegex(RuntimeEnvironmentError, "no module identities"):
            normalize_module_list("")

    def test_complete_document_and_stable_normalized_ldd(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            module_list = directory / "module-list.txt"
            module_list.write_bytes(b"runtime/1\r\nopenmpi/test\r\n")
            real_library_directory = directory / "runtime-real"
            real_library_directory.mkdir()
            real_library = real_library_directory / "libfixture.so"
            real_library.write_bytes(b"fixture library\n")
            alias_library_directory = directory / "runtime-alias"
            alias_library_directory.symlink_to(
                real_library_directory, target_is_directory=True
            )
            alias_library = alias_library_directory / real_library.name
            canonical_library = str(real_library.resolve())
            state = {"ldd": 0, "library_path": str(alias_library)}

            def which(name):
                return "/fake/" + name

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    state["ldd"] += 1
                    address = "0x0000{0}".format(state["ldd"])
                    return 0, (
                        "libfixture.so => {0} ({1})\r\n".format(
                            state["library_path"], address
                        )
                    )
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
            first, first_evidence = collect_runtime_environment_documents(
                inputs["manifest"], [inputs["metadata"]], module_list,
                str(directory / "cuda"), "13.0.88",
                str(directory / "cuda"), environment, execute, which,
                require_ldd=True, require_gpu_tools=True,
                cuda_runtime_probe=successful_cuda_runtime_probe,
            )
            state["library_path"] = canonical_library
            second, second_evidence = collect_runtime_environment_documents(
                inputs["manifest"], [inputs["metadata"]], module_list,
                str(directory / "cuda"), "13.0.88",
                str(directory / "cuda"), environment, execute, which,
                require_ldd=True, require_gpu_tools=True,
                cuda_runtime_probe=successful_cuda_runtime_probe,
            )
            self.assertEqual(dump_bytes(first), dump_bytes(second))
            self.assertEqual(
                deterministic_json_sha256(first),
                deterministic_json_sha256(second),
            )
            self.assertNotEqual(
                dump_bytes(first_evidence), dump_bytes(second_evidence)
            )
            self.assertIn(
                "(0x00001)",
                first_evidence["binaries"][0]["ldd"]["output_text"],
            )
            self.assertIn(
                "(0x00002)",
                second_evidence["binaries"][0]["ldd"]["output_text"],
            )
            self.assertIn(
                str(alias_library),
                first_evidence["binaries"][0]["ldd"]["output_text"],
            )
            self.assertIn(
                canonical_library,
                second_evidence["binaries"][0]["ldd"]["output_text"],
            )
            self.assertTrue(
                first_evidence["binaries"][0]["ldd"]["output_text"]
                .endswith("\r\n")
            )
            self.assertEqual(
                first_evidence["runtime_environment_sha256"],
                deterministic_json_sha256(first),
            )
            self.assertEqual(first["runtime_environment_schema_version"], 1)
            self.assertEqual(first["cpu_runtime_environment"], CPU_ENVIRONMENT)
            self.assertEqual(
                first["cpu_thread_semantics"],
                {"requested_threads": 48,
                 "serial_cpu_baseline_effective_threads": 1},
            )
            self.assertEqual(first["resolved_shared_library_paths"],
                             [canonical_library])
            self.assertEqual(first["binaries"][0]["ldd"]["output"],
                             ["libfixture.so => " + canonical_library])
            self.assertEqual(first["cuda"]["configured_toolkit_version"],
                             "13.0.88")
            self.assertEqual(first["cuda"]["runtime_version"], "13.0.96")
            self.assertEqual(first["cuda"]["runtime_version_source"],
                             "cudaRuntimeGetVersion")
            self.assertEqual(first["nvhpc"]["selected_cuda_toolkit"],
                             str((directory / "cuda").resolve()))
            self.assertEqual(first["module_list"],
                             ["runtime/1", "openmpi/test"])
            self.assertEqual(
                first_evidence["module_list_output"],
                "runtime/1\r\nopenmpi/test\r\n",
            )
            self.assertNotIn("gpus", first["nvidia"])
            self.assertTrue(all(
                item["status"] == "success"
                for item in first["cpu_library_versions"].values()
            ))

    def test_resolved_library_path_change_changes_identity_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            state = {"path": "/opt/runtime-v1/lib/libfixture.so"}

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    return 0, "libfixture.so => {0} (0x1)\n".format(
                        state["path"]
                    )
                if name == "pkg-config":
                    return 0, "1.2.3\n"
                if name == "nvidia-smi":
                    return 0, "fixture-value\n"
                return 0, name + " fixture version\n"

            first, _ = collect_fixture_documents(
                directory, inputs, execute
            )
            state["path"] = "/opt/runtime-v2/lib/libfixture.so"
            second, _ = collect_fixture_documents(
                directory, inputs, execute
            )
            self.assertNotEqual(
                deterministic_json_sha256(first),
                deterministic_json_sha256(second),
            )

    def test_cpu_library_version_change_changes_identity_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            state = {"version": "1.2.3"}

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    return 0, "libfixture.so => /lib/libfixture.so (0x1)\n"
                if name == "pkg-config":
                    return 0, state["version"] + "\n"
                if name == "nvidia-smi":
                    return 0, "fixture-value\n"
                return 0, name + " fixture version\n"

            first, _ = collect_fixture_documents(
                directory, inputs, execute
            )
            state["version"] = "1.2.4"
            second, _ = collect_fixture_documents(
                directory, inputs, execute
            )
            self.assertNotEqual(
                deterministic_json_sha256(first),
                deterministic_json_sha256(second),
            )

    def test_unresolved_ldd_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    return 0, "libmissing.so => not found\n"
                return 0, "fixture\n"

            with self.assertRaisesRegex(
                RuntimeEnvironmentError, "unresolved library"
            ):
                collect_fixture_documents(directory, inputs, execute)

    def test_gpu_uuid_is_evidence_only_and_does_not_change_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            state = {"uuid": "GPU-wave-0", "address": "0x1000"}

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    return 0, (
                        "libfixture.so => /lib/libfixture.so ({0})\n"
                        .format(state["address"])
                    )
                if name == "pkg-config":
                    return 0, "1.2.3\n"
                if name == "nvidia-smi":
                    if any("uuid,name" in item for item in arguments):
                        return 0, state["uuid"] + ", Fixture GPU\n"
                    return 0, "580.95.05\n"
                return 0, name + " fixture version\n"

            first, first_evidence = collect_fixture_documents(
                directory, inputs, execute
            )
            state.update({"uuid": "GPU-wave-1", "address": "0x2000"})
            second, second_evidence = collect_fixture_documents(
                directory, inputs, execute
            )
            self.assertEqual(
                deterministic_json_sha256(first),
                deterministic_json_sha256(second),
            )
            self.assertNotEqual(
                dump_bytes(first_evidence), dump_bytes(second_evidence)
            )
            self.assertNotIn("gpus", first["nvidia"])
            self.assertIn(
                "GPU-wave-0",
                first_evidence["command_probes"]
                ["nvidia_gpu_identity"]["output_text"],
            )

    def test_search_path_normalization_and_meaningful_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)

            def execute(arguments):
                name = Path(arguments[0]).name
                if name == "ldd":
                    return 0, "libfixture.so => /lib/libfixture.so (0x1)\n"
                if name == "pkg-config":
                    return 0, "1.2.3\n"
                if name == "nvidia-smi":
                    return 0, "fixture-value\n"
                return 0, name + " fixture version\n"

            first, _ = collect_fixture_documents(
                directory, inputs, execute,
                {"PATH": "/fake/bin:/usr/bin"},
            )
            duplicate, _ = collect_fixture_documents(
                directory, inputs, execute,
                {"PATH": "/fake/bin:/usr/bin:/fake/bin"},
            )
            changed_path, _ = collect_fixture_documents(
                directory, inputs, execute,
                {"PATH": "/different/bin:/usr/bin"},
            )
            changed_ld, _ = collect_fixture_documents(
                directory, inputs, execute,
                {"LD_LIBRARY_PATH": "/different/lib:/usr/lib"},
            )
            self.assertEqual(
                deterministic_json_sha256(first),
                deterministic_json_sha256(duplicate),
            )
            self.assertNotEqual(
                deterministic_json_sha256(first),
                deterministic_json_sha256(changed_path),
            )
            self.assertNotEqual(
                deterministic_json_sha256(first),
                deterministic_json_sha256(changed_ld),
            )

    def test_module_display_layout_normalizes_to_identity_order(self):
        columnar = (
            "Currently Loaded Modulefiles:\n"
            " 1) hpcx <aL>                  3) onemkl/2025.3.1\n"
            " 2) openmpi/runtime            4) cuda/13.0.2\n"
            "\nKey:\n<module-tag>  <aL>=auto-loaded\n"
        )
        one_per_line = (
            "hpcx\nopenmpi/runtime\nonemkl/2025.3.1\ncuda/13.0.2\n"
        )
        expected = [
            "hpcx", "openmpi/runtime", "onemkl/2025.3.1", "cuda/13.0.2"
        ]
        self.assertEqual(normalize_module_list(columnar), expected)
        self.assertEqual(normalize_module_list(one_per_line), expected)

    @unittest.skipUnless(
        sys.platform.startswith("linux") and shutil.which("ldd"),
        "Linux ldd is unavailable on this host",
    )
    def test_linux_repeated_ldd_has_stable_canonical_output(self):
        raw_outputs = []
        for _ in range(2):
            completed = subprocess.run(
                [shutil.which("ldd"), sys.executable],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            raw_outputs.append(completed.stdout)
        if raw_outputs[0] == raw_outputs[1]:
            self.skipTest("ldd load addresses did not vary on this Linux host")
        self.assertIn("(0x", raw_outputs[0])
        self.assertEqual(
            normalize_ldd_output(raw_outputs[0].splitlines()),
            normalize_ldd_output(raw_outputs[1].splitlines()),
        )

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
