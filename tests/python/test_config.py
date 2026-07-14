import copy
import unittest
from pathlib import Path

from gpu_suite.config import ConfigError, load_config, validate_config
from gpu_suite.strict_json import dump_bytes


ROOT = Path(__file__).resolve().parents[2]


def config_fixture():
    return copy.deepcopy(load_config(ROOT / "configs" / "pilot.json"))


class ConfigTests(unittest.TestCase):
    def test_scope_specific_counts(self):
        validated = validate_config(config_fixture())
        case = validated["benchmarks"]["cufft"]["cases"][0]
        self.assertEqual(case["scopes"]["compute"], {
            "repeat": 2, "trials": 1, "warmup": 1
        })
        self.assertEqual(case["scopes"]["end-to-end"], {
            "repeat": 1, "trials": 1, "warmup": 1
        })

    def test_missing_scope_is_rejected(self):
        config = config_fixture()
        del config["benchmarks"]["cufft"]["cases"][0]["scopes"]["compute"]
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_solver_repeat_is_one(self):
        config = config_fixture()
        config["benchmarks"]["cusolver"]["cases"][0]["scopes"]["compute"]["repeat"] = 2
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_exact_pilot_calibration_candidates(self):
        config = config_fixture()
        self.assertEqual(config["cpu_threads"], 48)
        expected = {
            "cufft": ([256], {"batch": 8}),
            "cublas": ([128], {}),
            "cusparse": ([4096], {}),
            "cusolver": ([64], {"nrhs": 2}),
            "curand": ([65536], {}),
            "thrust": ([65536], {}),
        }
        for benchmark, (sizes, extras) in expected.items():
            cases = config["benchmarks"][benchmark]["cases"]
            key = "nfft" if benchmark == "cufft" else "size"
            self.assertEqual([case["parameters"][key] for case in cases], sizes)
            for name, value in extras.items():
                self.assertTrue(all(case["parameters"][name] == value for case in cases))
            for case in cases:
                compute = case["scopes"]["compute"]
                end_to_end = case["scopes"]["end-to-end"]
                self.assertEqual(compute["warmup"], 1)
                self.assertEqual(compute["trials"], 1)
                self.assertEqual(compute["repeat"], 1 if benchmark == "cusolver" else 2)
                self.assertEqual(end_to_end, {"repeat": 1, "trials": 1, "warmup": 1})
        cufft_series = config["benchmarks"]["cufft"]["series"]
        serial = next(item for item in cufft_series if
                      item["cpu_backend"] == "cpu-fftw-serial")
        self.assertEqual(serial["series_role"], "auxiliary")
        self.assertEqual(
            config["benchmarks"]["cufft"]["default_speedup_cpu_backend"],
            "cpu-fftw-threaded",
        )
        verification = config["benchmarks"]["curand"]["verification"]
        self.assertEqual(verification["sigma_multiplier"], 6.0)
        self.assertEqual(verification["expected_mean"], 0.5)
        self.assertEqual(
            verification["expected_second_central_moment"], 1.0 / 12.0
        )

    def test_exact_benchmark_calibration_candidates(self):
        config = load_config(ROOT / "configs" / "benchmark.json")
        expected_sizes = {
            "cufft": [256, 1024, 4096, 16384],
            "cublas": [512, 1024, 2048, 4096],
            "cusparse": [65536, 262144, 1048576, 4194304],
            "cusolver": [256, 512, 1024, 2048],
            "curand": [1048576, 4194304, 16777216, 67108864],
            "thrust": [1048576, 4194304, 16777216, 67108864],
        }
        repeats = {
            "cufft": 3, "cublas": 3, "cusparse": 10,
            "cusolver": 1, "curand": 3, "thrust": 10,
        }
        for benchmark, sizes in expected_sizes.items():
            cases = config["benchmarks"][benchmark]["cases"]
            key = "nfft" if benchmark == "cufft" else "size"
            self.assertEqual([case["parameters"][key] for case in cases], sizes)
            for case in cases:
                self.assertEqual(case["scopes"]["compute"], {
                    "repeat": repeats[benchmark], "trials": 5, "warmup": 1
                })
                self.assertEqual(case["scopes"]["end-to-end"], {
                    "repeat": 1, "trials": 5, "warmup": 1
                })
        self.assertTrue(all(
            case["parameters"]["batch"] == 4096
            for case in config["benchmarks"]["cufft"]["cases"]
        ))
        self.assertTrue(all(
            case["parameters"]["nrhs"] == 16
            for case in config["benchmarks"]["cusolver"]["cases"]
        ))

    def test_configs_use_deterministic_json_profile(self):
        for name in ("pilot.json", "benchmark.json"):
            path = ROOT / "configs" / name
            self.assertEqual(path.read_bytes(), dump_bytes(load_config(path)))


if __name__ == "__main__":
    unittest.main()
