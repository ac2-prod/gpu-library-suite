import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gpu_suite.strict_json import dump_bytes, load

from pegasus_support import pegasus_config, write_campaign_inputs
from support import ROOT, ZERO_HASH, pilot_config


PEGASUS_DIRECTORY = ROOT / "jobs" / "pegasus"
sys.path.insert(0, str(PEGASUS_DIRECTORY))

from job_config import (  # noqa: E402
    CPU_ENVIRONMENT_KEYS,
    PegasusConfigError,
    load_pegasus_config,
    validate_pegasus_config,
)
from prepare_wave import (  # noqa: E402
    WavePreparationError,
    parse_rank_host_mapping,
    prepare_wave,
)
from render_job import render_job  # noqa: E402


class PegasusConfigurationAndRenderTests(unittest.TestCase):
    def test_renderer_requires_existing_pbs_output_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "results").mkdir()
            (directory / "scratch").mkdir()
            config_path = directory / "pegasus.json"
            config_path.write_bytes(dump_bytes(pegasus_config(directory)))
            inputs = write_campaign_inputs(directory)
            with self.assertRaisesRegex(ValueError, "parent directory"):
                render_job(
                    PEGASUS_DIRECTORY / "run_benchmarks.pbs.in", config_path,
                    ROOT, inputs["config"], inputs["manifest"],
                    [inputs["metadata"]], "render-run", 0, "pegasus-test",
                    "gpu-suite-test",
                )

    def test_rendered_job_has_separate_runtime_modules_and_all_cpu_variables(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("results", "scratch", "pbs"):
                (directory / name).mkdir()
            config_document = pegasus_config(directory)
            config_path = directory / "pegasus.json"
            config_path.write_bytes(dump_bytes(config_document))
            inputs = write_campaign_inputs(directory)
            rendered = render_job(
                PEGASUS_DIRECTORY / "run_benchmarks.pbs.in", config_path,
                ROOT, inputs["config"], inputs["manifest"], [inputs["metadata"]],
                "render-run", 0, "pegasus-test", "gpu-suite-test",
            )
            self.assertTrue(rendered.startswith("#!/bin/bash\n"))
            self.assertIn("module load runtime/1", rendered)
            self.assertNotIn("cpu-build/1", rendered)
            self.assertNotIn("openacc-build/1", rendered)
            for name in CPU_ENVIRONMENT_KEYS:
                self.assertIn("export {0}=".format(name), rendered)
            self.assertIn("GPU_SUITE_CPU_THREADS_REQUESTED=48", rendered)
            self.assertIn("GPU_SUITE_SERIAL_CPU_THREADS_EFFECTIVE=1", rendered)
            self.assertLess(rendered.index("prepare_wave.py"),
                            rendered.index("run_node.sh"))
            self.assertLess(rendered.index("run_node.sh"),
                            rendered.index("collect_results.py"))
            self.assertEqual(rendered.count("mpirun ${NQSV_MPIOPTS}"), 2)
            self.assertNotIn('mpirun "${NQSV_MPIOPTS}"', rendered)
            self.assertNotIn("--require-runtime-compilers", rendered)

            strict_config = dict(config_document)
            strict_config["require_runtime_compilers"] = True
            strict_path = directory / "strict-pegasus.json"
            strict_path.write_bytes(dump_bytes(strict_config))
            strict_rendered = render_job(
                PEGASUS_DIRECTORY / "run_benchmarks.pbs.in", strict_path,
                ROOT, inputs["config"], inputs["manifest"],
                [inputs["metadata"]], "render-run-strict", 0,
                "pegasus-test", "gpu-suite-test",
            )
            self.assertIn("--require-runtime-compilers", strict_rendered)
            rendered_path = directory / "rendered.pbs"
            rendered_path.write_text(rendered, encoding="utf-8")
            completed = subprocess.run(
                ["bash", "-n", str(rendered_path)], text=True,
                capture_output=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_missing_required_config_and_example_placeholders_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = pegasus_config(Path(temporary))
            del config["account"]
            with self.assertRaises(PegasusConfigError):
                validate_pegasus_config(config)
        example_path = PEGASUS_DIRECTORY / "pegasus.json.example"
        self.assertEqual(example_path.read_bytes(), dump_bytes(load(example_path)))
        with self.assertRaises(PegasusConfigError):
            load_pegasus_config(example_path)

    def test_cuda_toolkit_version_requires_exact_component_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = pegasus_config(Path(temporary))
            config["cuda_toolkit_version"] = "13.0"
            with self.assertRaisesRegex(
                PegasusConfigError, "CMake/NVCC component version"
            ):
                validate_pegasus_config(config)
            config["cuda_toolkit_version"] = "cuda/13.0.2"
            with self.assertRaisesRegex(
                PegasusConfigError, "CMake/NVCC component version"
            ):
                validate_pegasus_config(config)
            config["cuda_toolkit_version"] = "13.0.88"
            self.assertEqual(
                validate_pegasus_config(config)["cuda_toolkit_version"],
                "13.0.88",
            )

    def test_build_scripts_fix_profiles_and_module_groups(self):
        cpu_script = (PEGASUS_DIRECTORY / "build_cpu_cuda.sh").read_text(encoding="utf-8")
        openacc_script = (PEGASUS_DIRECTORY / "build_openacc.sh").read_text(encoding="utf-8")
        self.assertIn("modules --profile cpu-cuda", cpu_script)
        self.assertIn("GPU_SUITE_BUILD_CPU=ON", cpu_script)
        self.assertIn("GPU_SUITE_BUILD_CUDA=ON", cpu_script)
        self.assertIn("GPU_SUITE_BUILD_OPENACC=OFF", cpu_script)
        self.assertIn("modules --profile openacc", openacc_script)
        self.assertIn("CMAKE_C_COMPILER=nvc", openacc_script)
        self.assertIn("CMAKE_CXX_COMPILER=nvc++", openacc_script)
        self.assertIn("GPU_SUITE_BUILD_CPU=OFF", openacc_script)
        self.assertIn("GPU_SUITE_BUILD_CUDA=OFF", openacc_script)
        self.assertIn("GPU_SUITE_BUILD_OPENACC=ON", openacc_script)


class PrepareWaveTests(unittest.TestCase):
    def test_runtime_document_must_match_manifest_and_build_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_root = directory / "results"
            result_root.mkdir()
            inputs = write_campaign_inputs(directory)
            mapping = directory / "mapping.tsv"
            mapping.write_text("0\tnode0\n", encoding="utf-8")

            wrong_manifest = load(inputs["runtime"])
            wrong_manifest["executables_manifest_sha256"] = ZERO_HASH
            wrong_manifest_path = directory / "wrong-manifest-runtime.json"
            wrong_manifest_path.write_bytes(dump_bytes(wrong_manifest))
            with self.assertRaisesRegex(WavePreparationError, "different manifest"):
                prepare_wave(
                    result_root, "wrong-runtime-manifest", 0, 1, mapping,
                    inputs["config"], inputs["manifest"], [inputs["metadata"]],
                    wrong_manifest_path, "pegasus-test", "NQSV", "1.test",
                    "master0",
                )

            wrong_metadata = load(inputs["runtime"])
            wrong_metadata["build_metadata"][0]["sha256"] = ZERO_HASH
            wrong_metadata_path = directory / "wrong-metadata-runtime.json"
            wrong_metadata_path.write_bytes(dump_bytes(wrong_metadata))
            with self.assertRaisesRegex(WavePreparationError, "metadata hashes"):
                prepare_wave(
                    result_root, "wrong-runtime-metadata", 0, 1, mapping,
                    inputs["config"], inputs["manifest"], [inputs["metadata"]],
                    wrong_metadata_path, "pegasus-test", "NQSV", "2.test",
                    "master0",
                )

    def test_new_campaign_additional_wave_and_runtime_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_root = directory / "results"
            result_root.mkdir()
            inputs = write_campaign_inputs(directory)
            mapping = directory / "mapping.tsv"
            mapping.write_text("0\tnode0\n1\tnode1\n", encoding="utf-8")
            first = prepare_wave(
                result_root, "campaign-1", 0, 2, mapping, inputs["config"],
                inputs["manifest"], [inputs["metadata"]], inputs["runtime"],
                "pegasus-test", "NQSV", "123.test", "master0",
                timestamp="2026-07-14T00:00:00.000Z",
            )
            self.assertEqual(first["observed_node_count"], 2)
            self.assertEqual(first["permutation_assignment_counts"]["0"], 1)
            self.assertTrue((result_root / "campaign-1" / "run-metadata.json").is_file())
            second = prepare_wave(
                result_root, "campaign-1", 1, 2, mapping, inputs["config"],
                inputs["manifest"], [inputs["metadata"]], inputs["runtime"],
                "pegasus-test", "NQSV", "124.test", "master0",
                timestamp="2026-07-14T01:00:00.000Z",
            )
            self.assertEqual(second["wave"], 1)

            changed_runtime = directory / "changed-runtime.json"
            runtime_document = load(inputs["runtime"])
            runtime_document["test_runtime"] = "two"
            changed_runtime.write_bytes(dump_bytes(runtime_document))
            with self.assertRaisesRegex(WavePreparationError, "new run ID"):
                prepare_wave(
                    result_root, "campaign-1", 2, 2, mapping, inputs["config"],
                    inputs["manifest"], [inputs["metadata"]], changed_runtime,
                    "pegasus-test", "NQSV", "125.test", "master0",
                )

    def test_mapping_completeness_duplicates_single_node_and_wave_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            incomplete = directory / "incomplete.tsv"
            incomplete.write_text("0\tnode0\n", encoding="utf-8")
            with self.assertRaises(WavePreparationError):
                parse_rank_host_mapping(incomplete, 2)
            duplicate = directory / "duplicate.tsv"
            duplicate.write_text("0\tnode0\n1\tnode0\n", encoding="utf-8")
            with self.assertRaises(WavePreparationError):
                parse_rank_host_mapping(duplicate, 2)

            result_root = directory / "results"
            result_root.mkdir()
            inputs = write_campaign_inputs(directory)
            prepare_wave(
                result_root, "single", 0, 1, incomplete, inputs["config"],
                inputs["manifest"], [inputs["metadata"]], inputs["runtime"],
                "single-test", "NQSV", "1.test", "master0",
            )
            with self.assertRaisesRegex(WavePreparationError, "already exists"):
                prepare_wave(
                    result_root, "single", 0, 1, incomplete, inputs["config"],
                    inputs["manifest"], [inputs["metadata"]], inputs["runtime"],
                    "single-test", "NQSV", "2.test", "master0",
                )

    def test_production_rejects_dirty_build_even_with_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result_root = directory / "results"
            result_root.mkdir()
            config = pilot_config()
            config["run_mode"] = "production"
            inputs = write_campaign_inputs(directory, dirty=True, config=config)
            mapping = directory / "mapping.tsv"
            mapping.write_text("0\tnode0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_wave(
                    result_root, "dirty-production", 0, 1, mapping,
                    inputs["config"], inputs["manifest"], [inputs["metadata"]],
                    inputs["runtime"], "pegasus-test", "NQSV", "3.test",
                    "master0", source_snapshot_sha256=ZERO_HASH,
                )


if __name__ == "__main__":
    unittest.main()
