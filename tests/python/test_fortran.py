"""Fortran integration checks. GPU rows used below are explicitly synthetic."""
import copy
import hashlib
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

from gpu_suite.aggregation import aggregate_results
from gpu_suite.config import load_config, validate_config, ConfigError
from gpu_suite.manifest import merge_entries, artifact_id, ManifestError
from gpu_suite.plotting import primary_plot_series, PlotError, build_plot_metadata, render_plots
from gpu_suite.results_io import parse_stdout_prefix, load_raw_results
from gpu_suite.runner import (build_schedule, load_manifest, load_build_metadata,
                             validate_manifest_artifacts, execute_schedule,
                             expected_raw_parameters, core_raw_parameters, RunnerError)
from gpu_suite.schema import validate_raw_result, SchemaError
from gpu_suite.strict_json import dump_bytes, loads
from gpu_suite.validation import validate_campaign
from support import ROOT, ZERO_HASH, pilot_config, raw_success
from test_plotting import publication_records, raw_version_records


class FortranSourceTests(unittest.TestCase):
    def test_eighteen_examples_and_benchmarks_are_real_fortran(self):
        root = ROOT / "nvidia/fortran"
        self.assertEqual(len(list(root.glob("*/examples/*.f90"))), 18)
        self.assertEqual(len(list(root.glob("*/benchmarks/*_bench.f90"))), 18)
        self.assertEqual(len(list(root.glob("*/benchmarks/*_workload.F90"))), 6)
        for source in list(root.rglob("*.f90")) + list(root.rglob("*.F90")):
            text = source.read_text()
            self.assertNotIn("execute_command_line", text)
            self.assertNotIn("...", text)
            self.assertTrue(all(len(line) <= 132 for line in text.splitlines()), source)
        self.assertIn("call dgemm", (root / "cublas/benchmarks/blas_workload.F90").read_text())
        self.assertIn("call random_number", (root / "curand/benchmarks/rand_workload.F90").read_text())
        self.assertIn("sum(values*values)", (root / "thrust/benchmarks/reduce_workload.F90").read_text())

    def test_language_config_is_optional_and_strict(self):
        old = pilot_config()
        self.assertEqual(validate_config(old), old)
        new = load_config(ROOT / "configs/fortran/pilot.json")
        self.assertEqual(new["source_language"], "fortran")
        for invalid in ("unknown", None, 1, [], {}):
            new["source_language"] = invalid
            with self.assertRaises(ConfigError):
                validate_config(new)

    def test_parameter_language_separates_aggregation(self):
        old = raw_success(benchmark="cublas")
        new = copy.deepcopy(old)
        new["parameters"]["source_language"] = "fortran"
        rows, _ = aggregate_results([old, new], pilot_config())
        blocks = [row for row in rows if row["summary_level"] == "block" and row["comparison"] is None]
        self.assertEqual(len(blocks), 2)
        self.assertNotEqual(blocks[0]["parameter_signature"], blocks[1]["parameter_signature"])
        with self.assertRaises(PlotError):
            primary_plot_series(rows)

    def test_intrinsic_rng_identity_is_not_cpp(self):
        config = load_config(ROOT / "configs/fortran/pilot.json")
        params = config["benchmarks"]["curand"]["cases"][0]["parameters"]
        actual = expected_raw_parameters("curand", params, "cpu", "fortran")
        self.assertEqual(actual["cpu_engine"], "Fortran random_number")
        self.assertEqual(core_raw_parameters("curand", actual)["source_language"], "fortran")

    def test_raw_language_rejects_unknown_values(self):
        row = raw_success()
        row["parameters"]["source_language"] = "unexpected"
        with self.assertRaises(SchemaError):
            validate_raw_result(row)

    def test_synthetic_six_library_plot_labels(self):
        # No timing values in this fixture are GPU measurements.
        records = publication_records()
        for row in records:
            row["parameters"] = {"source_language": "fortran"}
            if row["benchmark"] == "curand" and row["implementation"] == "cpu":
                row["cpu_backend"] = "cpu-fortran-random-serial"
            if row["benchmark"] == "thrust" and row["implementation"] == "cpu":
                row["cpu_backend"] = "cpu-fortran-sum-serial"
        series = primary_plot_series(records)
        self.assertEqual({row["benchmark"] for row in series},
                         {"cufft", "cublas", "cusparse", "cusolver", "curand", "thrust"})
        self.assertTrue(all("Fortran" in row["label"] for row in series))
        self.assertTrue(any("random_number" in row["label"] for row in series))
        self.assertTrue(any("Fortran sum" in row["label"] for row in series))
        with self.assertRaisesRegex(PlotError, "source languages differ"):
            build_plot_metadata(records, ZERO_HASH, raw_records=raw_version_records())


@unittest.skipUnless(os.environ.get("GPU_SUITE_FORTRAN_BUILD"), "Fortran build is not supplied")
class FortranRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = Path(os.environ["GPU_SUITE_FORTRAN_BUILD"])
        cls.manifest, _ = load_manifest(cls.build / "partial-manifest.json")
        cls.metadata = load_build_metadata([cls.build / "build-metadata.json"])

    def command(self, library, stem, scope, output_format="jsonl", extra=()):
        executable = self.build / "nvidia/fortran" / library / (stem + "_cpu_bench")
        return [str(executable), "--size", "65536", "--scope", scope,
                "--warmup", "2", "--repeat", "2", "--trials", "3", "--verify", "true",
                "--output", "-", "--format", output_format] + list(extra)

    def test_intrinsic_compute_e2e_restoration_json_and_csv(self):
        for library, stem in (("curand", "rand"), ("thrust", "reduce")):
            for scope in ("compute", "end-to-end"):
                for output_format in ("jsonl", "csv"):
                    with self.subTest(library=library, scope=scope, format=output_format):
                        run = subprocess.run(self.command(library, stem, scope, output_format),
                                             text=True, capture_output=True, check=False)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        rows, error = parse_stdout_prefix(run.stdout, output_format)
                        self.assertIsNone(error)
                        self.assertEqual(len(rows), 3)
                        for row in rows:
                            validate_raw_result(row)
                            self.assertEqual(row["verification_status"], "pass")
                            self.assertEqual(row["parameters"]["source_language"], "fortran")
                            self.assertEqual(row["cpu_threads_effective"], 1)
                        self.assertEqual(rows[0]["verification_metrics"], rows[1]["verification_metrics"])
                        self.assertEqual(rows[1]["verification_metrics"], rows[2]["verification_metrics"])

    def test_wrong_backend_and_overflow_are_failures(self):
        run = subprocess.run(self.command("curand", "rand", "compute", extra=("--cpu-backend", "cpu-std-random-serial")),
                             text=True, capture_output=True)
        self.assertNotEqual(run.returncode, 0)
        command = self.command("thrust", "reduce", "compute")
        command[command.index("--size")+1] = "2147483648"
        run = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(run.returncode, 0)
        rows, error = parse_stdout_prefix(run.stdout, "jsonl")
        self.assertIsNone(error)
        self.assertEqual([row["status"] for row in rows], ["failure", "skipped", "skipped"])

    def test_synthetic_fortran_six_figure_render(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is unavailable; no PNG rendering claim")
        records, raw = publication_records(), raw_version_records()
        for row in records + raw:
            row.setdefault("parameters", {})["source_language"] = "fortran"
            if row["implementation"] == "cpu" and row["benchmark"] in {"curand", "thrust"}:
                row["cpu_backend"] = ("cpu-fortran-random-serial" if row["benchmark"] == "curand"
                                      else "cpu-fortran-sum-serial")
        metadata = build_plot_metadata(records, ZERO_HASH, raw_records=raw)
        directory = Path(tempfile.mkdtemp(prefix="synthetic-fortran-plots-", dir=str(self.build)))
        rendered = render_plots(metadata, directory)
        self.assertEqual(rendered["matplotlib"]["status"], "rendered")
        self.assertEqual(len(rendered["generated_files"]), 6)
        for filename in rendered["generated_files"]:
            self.assertTrue((directory / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        print("Synthetic Fortran plot test output (NOT measurements):", directory)

    def test_metadata_manifest_hashes_and_language_gate(self):
        content = (self.build / "build-metadata.json").read_bytes()
        self.assertEqual(content, dump_bytes(loads(content)))
        digest = hashlib.sha256(content).hexdigest()
        for entry in self.manifest["entries"]:
            self.assertEqual(entry["compiler_language"], "fortran")
            self.assertEqual(entry["build_metadata_sha256"], digest)
            self.assertIn("-O3", entry["global_configure_flags"])
            self.assertNotIn("-fast ", entry["global_configure_flags"])
        validate_manifest_artifacts(self.manifest, self.metadata, "pilot", True)
        with self.assertRaises(RunnerError):
            build_schedule(pilot_config(), self.manifest, {})
        mixed = copy.deepcopy(self.manifest["entries"][0])
        mixed["compiler_language"] = "c"
        mixed["library"] = "cublas"
        mixed["artifact_id"] = artifact_id(mixed)
        with self.assertRaises(ManifestError):
            merge_entries([self.manifest["entries"], [mixed]])

    def test_runner_validation_and_aggregation_with_unavailable_gpu(self):
        config = load_config(ROOT / "configs/fortran/pilot.json")
        for name, definition in config["benchmarks"].items():
            definition["enabled"] = name in {"curand", "thrust"}
            definition["cases"] = definition["cases"][:1]
        context = {"config": config, "cpu_threads": config["cpu_threads"], "device": 0,
                   "hostname": socket.gethostname(), "implementation_order": ("cpu", "cuda", "openacc"),
                   "node_index": 0, "run_id": "fortran-test", "scheduler": None, "scheduler_job_id": None,
                   "system_label": "local-test", "wave": 0}
        schedule = build_schedule(config, self.manifest, context)
        digest = hashlib.sha256(dump_bytes(config)).hexdigest()
        # The all-zero provenance below is test-only, never a measurement campaign.
        with tempfile.TemporaryDirectory(prefix="fortran-runner-", dir=str(self.build)) as directory:
            path = Path(directory) / "synthetic-context-raw.jsonl"
            status = execute_schedule(schedule, context, path, digest, ZERO_HASH,
                                      self.metadata, ZERO_HASH, None)
            self.assertEqual(status, 1)  # CUDA/OpenACC missing is not success.
            rows = load_raw_results(path)
            self.assertEqual(sum(row["status"] == "success" for row in rows), 8)
            self.assertTrue(all(row["status"] == "skipped" for row in rows if row["implementation"] != "cpu"))
            validate_campaign(rows, config, digest, self.manifest)
            aggregate, _ = aggregate_results(rows, config)
            self.assertTrue(all(row["parameters"]["source_language"] == "fortran" for row in aggregate))


if __name__ == "__main__":
    unittest.main()
