"""Six-family data contracts; no GPU, SSH or scheduler execution."""
import copy
from pathlib import Path
import unittest

from gpu_suite.config import load_config, validate_config
from gpu_suite.families import AMD_BENCHMARKS, AMD_COUNTERPARTS
from gpu_suite.manifest import artifact_id, merge_entries
from gpu_suite.plotting import build_plot_metadata
from gpu_suite.runner import (_expected_verification, build_schedule, core_raw_parameters,
                              cpu_threads_effective_matches, expected_raw_parameters,
                              normalized_parameters, validate_subprocess_record)
from gpu_suite.schema import validate_raw_result
from gpu_suite.validation import validate_campaign, validate_rocfft_provenance
from test_rocfft_tools import ZERO, amd_entry, campaign

ROOT = Path(__file__).resolve().parents[2]


class AmdFamiliesTests(unittest.TestCase):
    def test_six_library_smoke_config_and_solver_repeat(self):
        config = load_config(ROOT / "configs/amd/smoke.json")
        self.assertEqual(set(config["benchmarks"]), set(AMD_BENCHMARKS))
        self.assertEqual(config["vendor"], "amd")
        bad = copy.deepcopy(config)
        bad["benchmarks"]["rocsolver"]["cases"][0]["scopes"]["compute"]["repeat"] = 2
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_amd_subsets_are_explicit_and_cross_vendor_is_rejected(self):
        config = load_config(ROOT / "configs/amd/smoke.json")
        for library in AMD_BENCHMARKS:
            subset = dict(config, benchmarks={library: config["benchmarks"][library]})
            self.assertEqual(set(validate_config(subset)["benchmarks"]), {library})
            wrong = copy.deepcopy(subset)
            wrong["benchmarks"][AMD_COUNTERPARTS[library]] = wrong["benchmarks"].pop(library)
            with self.assertRaises(ValueError):
                validate_config(wrong)
        with self.assertRaises(ValueError):
            validate_config(dict(config, benchmarks={}))

    def test_full_manifest_keeps_library_and_implementation_identity(self):
        stems = dict(zip(AMD_BENCHMARKS, ("fft", "blas", "sparse", "solver", "rand", "reduce")))
        entries = []
        for library in AMD_BENCHMARKS:
            for impl in ("cpu", "hip", "openmp"):
                entry = amd_entry(impl)
                stem = "openmp_" + library if impl == "openmp" else stems[library] + ("_cpu" if impl == "cpu" else "_gpu")
                entry.update(library=library, target_name=stem + "_bench",
                             executable_path="/fixture/" + stem + "_bench")
                entry["artifact_id"] = artifact_id(entry)
                entries.append(entry)
        self.assertEqual(len(merge_entries([entries])), 18)
        wrong = copy.deepcopy(entries)
        wrong[4]["implementation"] = "cuda"
        wrong[4]["artifact_id"] = artifact_id(wrong[4])
        with self.assertRaises(ValueError):
            merge_entries([wrong])

    def test_parameters_keep_mathematics_but_not_runtime(self):
        config = load_config(ROOT / "configs/amd/smoke.json")
        for library, definition in config["benchmarks"].items():
            params = definition["cases"][0]["parameters"]
            expected = normalized_parameters(library, params)
            self.assertEqual(expected, normalized_parameters(AMD_COUNTERPARTS[library], params))
            self.assertEqual(core_raw_parameters(library, dict(expected, runtime={"ignored": True})), expected)

    def test_rocrand_algorithm_name_and_distribution_are_not_cuda(self):
        params = load_config(ROOT / "configs/amd/smoke.json")["benchmarks"]["rocrand"]["cases"][0]["parameters"]
        gpu = expected_raw_parameters("rocrand", params, "hip")
        self.assertEqual(gpu["generator_algorithm"], "ROCRAND_RNG_PSEUDO_DEFAULT")
        self.assertEqual(gpu["distribution_interval"], "(0,1]")
        cpu = expected_raw_parameters("rocrand", params, "cpu")
        self.assertEqual(cpu["cpu_engine"], "std::mt19937_64")
        self.assertEqual(cpu["distribution_interval"], "[0,1)")

    def test_unsupported_rng_and_reduction_algorithms_stop_at_config(self):
        config = load_config(ROOT / "configs/amd/smoke.json")
        for library, parameter, value in (
            ("rocrand", "generator", "xorwow"),
            ("rocrand", "distribution", "normal"),
            ("rocrand", "order", "seeded"),
            ("rocthrust", "operation", "sum"),
        ):
            changed = copy.deepcopy(config)
            changed["benchmarks"][library]["cases"][0]["parameters"][parameter] = value
            with self.subTest(library=library, parameter=parameter), self.assertRaises(ValueError):
                validate_config(changed)

    def test_unknown_effective_threads_accept_observation_in_runner_and_campaign(self):
        config, manifest, context, schedule, records = campaign()
        for series in config["benchmarks"]["rocfft"]["series"]:
            if series["implementation"] == "cpu":
                series["cpu_threads_effective"] = None
        validate_config(config)
        self.assertEqual(validate_campaign(records, config, ZERO, manifest)["validation_status"], "pass")
        item = next(item for item in schedule if item["series"]["implementation"] == "cpu")
        row = next(copy.deepcopy(row) for row in records if row["implementation"] == "cpu" and row["scope"] == item["scope"])
        thresholds, metrics, primary = _expected_verification(item, context)
        row.update(verification_thresholds=thresholds,
                   verification_metrics={metric: 0.0 for metric in metrics},
                   verification_primary_metric=primary)
        validate_subprocess_record(row, item, context, ZERO, ZERO, ZERO, None)
        item["series"]["cpu_threads_effective"] = 2
        with self.assertRaises(ValueError):
            validate_subprocess_record(row, item, context, ZERO, ZERO, ZERO, None)
        for series in config["benchmarks"]["rocfft"]["series"]:
            if series["implementation"] == "cpu":
                series["cpu_threads_effective"] = 2
        with self.assertRaises(ValueError):
            validate_campaign(records, config, ZERO, manifest)

    def test_rng_and_reduction_thread_observations_are_not_invented(self):
        row = next(copy.deepcopy(row) for row in campaign()[4] if row["implementation"] == "cpu")
        series = {"cpu_threads_effective": None}
        for library in ("rocrand", "rocthrust"):
            row.update(benchmark=library, cpu_parallelism="serial", cpu_threads_effective=1)
            self.assertTrue(cpu_threads_effective_matches(row, series))
            for invalid in (None, 2):
                row["cpu_threads_effective"] = invalid
                self.assertFalse(cpu_threads_effective_matches(row, series))
        row.update(benchmark="rocthrust", cpu_backend="cpu-openmp", cpu_parallelism="threaded",
                   cpu_threads_requested=4, cpu_threads_effective=4)
        self.assertTrue(cpu_threads_effective_matches(row, series))
        for invalid in (None, 1, 8):
            row["cpu_threads_effective"] = invalid
            self.assertFalse(cpu_threads_effective_matches(row, series))
        row.update(status="skipped", cpu_threads_effective=None)
        self.assertTrue(cpu_threads_effective_matches(row, series))

    @staticmethod
    def _amd_plot_fixture(library):
        from test_plotting import publication_records, raw_version_records
        counterpart = AMD_COUNTERPARTS[library]
        aggregates = [copy.deepcopy(row) for row in publication_records() if row["benchmark"] == counterpart]
        raw = [copy.deepcopy(row) for row in raw_version_records() if row["benchmark"] == counterpart]
        for row in aggregates + raw:
            row["benchmark"] = library
            row["implementation"] = {"cpu": "cpu", "cuda": "hip", "openacc": "openmp"}[row["implementation"]]
        return aggregates, raw

    def test_rocthrust_plot_requires_matching_hip_openmp_version_evidence(self):
        aggregates, raw = self._amd_plot_fixture("rocthrust")
        self.assertEqual(len(build_plot_metadata(aggregates, ZERO, raw)["series"]), 6)
        for absent in (None, []):
            with self.subTest(evidence=absent), self.assertRaises(ValueError):
                build_plot_metadata(aggregates, ZERO, absent)
        for mutation in ("mismatch", "missing", "multiple", "provenance"):
            altered = copy.deepcopy(raw)
            row = next(row for row in altered if row["implementation"] == "openmp")
            if mutation == "mismatch":
                for record in altered:
                    if record["implementation"] == "openmp": record["library_version"] = "different"
            elif mutation == "missing":
                altered = [record for record in altered if record["implementation"] != "openmp"]
            elif mutation == "multiple": row["library_version"] = "different"
            else: row["runtime_environment_sha256"] = "a" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_plot_metadata(aggregates, ZERO, altered)

    def test_rocrand_plot_keeps_algorithm_comparison_note(self):
        aggregates, raw = self._amd_plot_fixture("rocrand")
        metadata = build_plot_metadata(aggregates, ZERO, raw)
        self.assertIn("not algorithm-equivalent", metadata["rocrand_comparison_note"])

    def test_all_missing_artifacts_remain_explicit_prerequisites(self):
        config = load_config(ROOT / "configs/amd/smoke.json")
        context = campaign()[2]
        context["config"] = config
        schedule = build_schedule(config, {"entries": []}, context)
        self.assertEqual(len(schedule), 36)
        self.assertTrue(all(item["prerequisite_failure"] is not None for item in schedule))
        self.assertTrue(all(item["entry"] is None for item in schedule))

    def test_library_artifacts_may_differ_but_same_artifact_may_not(self):
        # Provenance gate alone: numerical schemas are checked independently.
        baseline = campaign()[4]
        first = next(copy.deepcopy(r) for r in baseline if r["implementation"] == "hip")
        other = copy.deepcopy(first)
        other.update(benchmark="rocblas", library_name="rocblas", library_version="blas-fixture", binary_sha256="a" * 64)
        other["parameters"]["runtime"]["build_metadata_sha256"] = "b" * 64
        validate_rocfft_provenance([first, other])
        changed = copy.deepcopy(other)
        changed["binary_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            validate_rocfft_provenance([first, other, changed])

    def test_cross_library_campaign_runtime_and_source_must_match(self):
        first = next(copy.deepcopy(r) for r in campaign()[4] if r["implementation"] == "hip")
        for field in ("source_snapshot_sha256", "config_sha256", "runtime_environment_sha256"):
            other = copy.deepcopy(first)
            other["benchmark"] = "rocblas"
            other[field] = "a" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rocfft_provenance([first, other])
        other = copy.deepcopy(first)
        other["benchmark"] = "rocblas"
        other["parameters"]["runtime"]["hip_runtime_version"] += 1
        with self.assertRaises(ValueError):
            validate_rocfft_provenance([first, other])

    def test_new_gpu_family_rejects_cuda_metadata_and_wrong_precision(self):
        row = next(copy.deepcopy(r) for r in campaign()[4] if r["implementation"] == "hip")
        row.update(benchmark="rocblas", precision="fp64", library_name="rocblas")
        validate_raw_result(row)
        for field, value in (("cuda_runtime_version", "12.0"), ("precision", "fp32")):
            bad = copy.deepcopy(row)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_raw_result(bad)


if __name__ == "__main__":
    unittest.main()
