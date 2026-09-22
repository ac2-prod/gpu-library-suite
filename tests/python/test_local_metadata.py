import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gpu_suite.hashing import sha256_file
from gpu_suite.strict_json import load
from pegasus_support import CPU_ENVIRONMENT, write_campaign_inputs
from support import ROOT

sys.path.insert(0, str(ROOT / "jobs" / "pegasus"))
from node_tools import NodeToolError, build_local_node_metadata
from prepare_wave import WavePreparationError, prepare_wave
from test_pegasus_runtime_telemetry import successful_cuda_runtime_probe


def execute_fixture(arguments):
    if Path(arguments[0]).name == "lscpu":
        return 0, '{"lscpu":[{"field":"Model name:","data":"Fixture CPU"}]}\n'
    return 0, "Fixture GPU, GPU-fixture, fixture-driver\n"


def local_fixture(config, manifest, run_id, wave, runtime_hash, gpu_uuid=None):
    return build_local_node_metadata(
        config, manifest, run_id, wave, runtime_hash, gpu_uuid,
        dict(CPU_ENVIRONMENT, PBS_JOBID="not-used"), execute_fixture,
        lambda name: "/fixture/" + name, successful_cuda_runtime_probe,
    )


class LocalMetadataTests(unittest.TestCase):
    def test_local_node_observes_identity_without_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = write_campaign_inputs(Path(temporary))
            node = local_fixture(inputs["config"], inputs["manifest"], "local-test", 0,
                                 sha256_file(inputs["runtime"]))
            self.assertIsNone(node["scheduler"])
            self.assertIsNone(node["scheduler_job_id"])
            self.assertEqual(node["hostname"], socket.gethostname())
            self.assertEqual(node["cpu_identity"], {
                "name": "Fixture CPU", "query_status": "success", "source": "lscpu",
            })
            self.assertEqual(node["gpu_identity"]["name"], "Fixture GPU")
            self.assertEqual(node["gpu_identity"]["uuid"], "GPU-fixture")
            self.assertEqual(node["execution_mode"], "local")

    def test_unavailable_cpu_is_recorded_not_invented(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = write_campaign_inputs(Path(temporary))
            node = build_local_node_metadata(
                inputs["config"], inputs["manifest"], "local-test", 0, "0" * 64,
                environment=CPU_ENVIRONMENT, which=lambda name: None,
                cuda_runtime_probe=successful_cuda_runtime_probe,
            )
            self.assertIsNone(node["cpu_identity"]["name"])
            self.assertEqual(node["cpu_identity"]["source"], "unknown")
            self.assertIsNone(node["gpu_identity"]["name"])

    def test_multiple_gpus_require_unambiguous_visibility(self):
        def multiple(arguments):
            if Path(arguments[0]).name == "lscpu":
                return execute_fixture(arguments)
            return 0, "Fixture A, GPU-a, driver\nFixture B, GPU-b, driver\n"
        with tempfile.TemporaryDirectory() as temporary:
            inputs = write_campaign_inputs(Path(temporary))
            args = (inputs["config"], inputs["manifest"], "local-test", 0, "0" * 64)
            kwargs = dict(environment=CPU_ENVIRONMENT, executor=multiple,
                          which=lambda name: "/fixture/" + name,
                          cuda_runtime_probe=successful_cuda_runtime_probe)
            with self.assertRaisesRegex(NodeToolError, "multiple local GPUs"):
                build_local_node_metadata(*args, **kwargs)
            with self.assertRaisesRegex(NodeToolError, "CUDA_VISIBLE_DEVICES"):
                build_local_node_metadata(*args, gpu_uuid="GPU-b", **kwargs)
            kwargs["environment"] = dict(CPU_ENVIRONMENT, CUDA_VISIBLE_DEVICES="GPU-b")
            node = build_local_node_metadata(*args, gpu_uuid="GPU-b", **kwargs)
            self.assertEqual(node["gpu_identity"]["uuid"], "GPU-b")

    def test_local_preparation_creates_real_run_wave_node_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            before = {name: sha256_file(path) for name, path in inputs.items()}
            output = directory / "results"
            output.mkdir()
            args = (
                output, "local-test", 0, 1, None, inputs["config"],
                inputs["manifest"], [inputs["metadata"]], inputs["runtime"],
                inputs["runtime_evidence"], "fixture-local", None, None, socket.gethostname(),
            )
            with mock.patch("prepare_wave.build_local_node_metadata", side_effect=local_fixture):
                wave = prepare_wave(*args, local=True)
                with self.assertRaisesRegex(WavePreparationError, "wave output already exists"):
                    prepare_wave(*args, local=True)
            run_root = output / "local-test"
            run = load(run_root / "run-metadata.json")
            self.assertEqual(run["launcher"]["execution_mode"], "local")
            self.assertIsNone(wave["scheduler"])
            self.assertIsNone(wave["scheduler_job_id"])
            self.assertEqual(wave["rank_host_mapping"], [
                {"hostname": socket.gethostname(), "rank": 0},
            ])
            wave_root = run_root / "waves" / "0"
            node = load(wave_root / "nodes" / socket.gethostname() / "node-metadata.json")
            self.assertEqual(node["runtime_environment_sha256"], sha256_file(inputs["runtime"]))
            self.assertEqual(sha256_file(wave_root / "runtime-environment-evidence.json"),
                             sha256_file(inputs["runtime_evidence"]))
            self.assertEqual(before, {name: sha256_file(path) for name, path in inputs.items()})

    def test_local_mode_rejects_fabricated_scheduler_and_changed_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            output = directory / "results"
            output.mkdir()
            args = [output, "local-test", 0, 1, None, inputs["config"], inputs["manifest"],
                    [inputs["metadata"]], inputs["runtime"], inputs["runtime_evidence"],
                    "fixture-local", "NQSV", "123.test", socket.gethostname()]
            with self.assertRaisesRegex(WavePreparationError, "no scheduler"):
                prepare_wave(*args, local=True)
            args[11:13] = [None, None]
            def changed(*values):
                node = local_fixture(*values)
                node["cpu_runtime_environment"]["OMP_NUM_THREADS"] = "7"
                return node
            with mock.patch("prepare_wave.build_local_node_metadata", side_effect=changed):
                with self.assertRaisesRegex(WavePreparationError, "CPU environment changed"):
                    prepare_wave(*args, local=True)
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
