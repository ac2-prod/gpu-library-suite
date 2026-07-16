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
            "repeat": 5197, "trials": 1, "warmup": 1
        })
        self.assertEqual(case["scopes"]["end-to-end"], {
            "repeat": 1, "trials": 1, "warmup": 1
        })

    def test_missing_scope_is_rejected(self):
        config = config_fixture()
        del config["benchmarks"]["cufft"]["cases"][0]["scopes"]["compute"]
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_missing_production_verification_operand_is_rejected(self):
        config = config_fixture()
        del config["benchmarks"]["cublas"]["verification"]["rel_tolerance"]
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_solver_repeat_is_one(self):
        config = config_fixture()
        config["benchmarks"]["cusolver"]["cases"][0]["scopes"]["compute"]["repeat"] = 2
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_exact_pilot_publication_matrix(self):
        config = config_fixture()
        self.assertEqual(config["cpu_threads"], 48)
        expected = {
            "cufft": ([256, 4096, 16384], [5197, 328, 83], {"batch": 4096}),
            "cublas": ([512, 2048, 4096], [2184, 133, 18], {}),
            "cusparse": ([65536, 1048576, 4194304], [5776, 1520, 206], {}),
            "cusolver": ([4096, 8192, 12288], [1, 1, 1], {"nrhs": 16}),
            "curand": ([1048576, 16777216, 67108864], [640, 173, 54], {}),
            "thrust": ([1048576, 16777216, 67108864], [1932, 532, 167], {}),
        }
        for benchmark, (sizes, repeats, extras) in expected.items():
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
                self.assertEqual(end_to_end, {"repeat": 1, "trials": 1, "warmup": 1})
            self.assertEqual(
                [case["scopes"]["compute"]["repeat"] for case in cases],
                repeats,
            )
        cufft_series = config["benchmarks"]["cufft"]["series"]
        self.assertEqual(len(cufft_series), 3)
        self.assertFalse(any(
            item["cpu_backend"] == "cpu-fftw-serial" for item in cufft_series
        ))
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
        self.assertEqual(self.configured_row_count(config), 108)

    def test_exact_benchmark_publication_matrix(self):
        config = load_config(ROOT / "configs" / "benchmark.json")
        expected = {
            "cufft": ([256, 4096, 16384], [5197, 328, 83]),
            "cublas": ([512, 2048, 4096], [2184, 133, 18]),
            "cusparse": ([65536, 1048576, 4194304], [5776, 1520, 206]),
            "cusolver": ([4096, 8192, 12288], [1, 1, 1]),
            "curand": ([1048576, 16777216, 67108864], [640, 173, 54]),
            "thrust": ([1048576, 16777216, 67108864], [1932, 532, 167]),
        }
        for benchmark, (sizes, repeats) in expected.items():
            cases = config["benchmarks"][benchmark]["cases"]
            key = "nfft" if benchmark == "cufft" else "size"
            self.assertEqual([case["parameters"][key] for case in cases], sizes)
            for case, repeat in zip(cases, repeats):
                self.assertEqual(case["scopes"]["compute"], {
                    "repeat": repeat, "trials": 5, "warmup": 1
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
        self.assertEqual(self.configured_row_count(config, node_wave_blocks=12), 6480)

    def test_expected_counts_when_thrust_is_excluded(self):
        pilot = config_fixture()
        pilot["benchmarks"]["thrust"]["enabled"] = False
        self.assertEqual(self.configured_row_count(pilot), 90)
        production = load_config(ROOT / "configs" / "benchmark.json")
        production["benchmarks"]["thrust"]["enabled"] = False
        self.assertEqual(
            self.configured_row_count(production, node_wave_blocks=12), 5400
        )

    @staticmethod
    def configured_row_count(config, node_wave_blocks=1):
        rows = 0
        for definition in config["benchmarks"].values():
            if not definition["enabled"]:
                continue
            for case in definition["cases"]:
                for scope in case["scopes"].values():
                    rows += len(definition["series"]) * scope["trials"]
        return rows * node_wave_blocks

    def test_configs_use_deterministic_json_profile(self):
        for name in ("pilot.json", "benchmark.json"):
            path = ROOT / "configs" / name
            self.assertEqual(path.read_bytes(), dump_bytes(load_config(path)))


if __name__ == "__main__":
    unittest.main()
