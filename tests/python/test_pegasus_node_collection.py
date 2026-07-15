import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gpu_suite.strict_json import dump_bytes, dumps, load

from pegasus_support import CPU_ENVIRONMENT, write_campaign_inputs
from support import ROOT, raw_success


PEGASUS_DIRECTORY = ROOT / "jobs" / "pegasus"
sys.path.insert(0, str(PEGASUS_DIRECTORY))

from collect_results import collect_results  # noqa: E402
from node_tools import (  # noqa: E402
    NodeToolError,
    build_node_metadata,
    build_node_status,
    classify_raw,
)


def write_wave_metadata(path, mapping):
    path.write_bytes(dump_bytes({
        "rank_host_mapping": mapping,
        "run_id": "run-1",
        "wave": 0,
        "wave_metadata_schema_version": 1,
    }))


def write_node(nodes, hostname, rank, classification):
    node = nodes / hostname
    node.mkdir()
    raw = raw_success(node_index=rank, hostname=hostname)
    (node / "raw-results.jsonl").write_text(dumps(raw) + "\n", encoding="utf-8")
    (node / "logs").mkdir()
    (node / "logs" / "runner.log").write_text("preserved\n", encoding="utf-8")
    status = build_node_status(
        "run-1", 0, rank, hostname,
        0 if classification["benchmark_status"] == "success" else 1,
        0 if classification["benchmark_status"] == "success" else 1,
        "success", "success", "success", "success", classification,
        {"telemetry_status": "success"}, [], None,
    )
    (node / "node-status.json").write_bytes(dump_bytes(status))
    return node / "raw-results.jsonl"


class CollectionTests(unittest.TestCase):
    def test_node_metadata_uses_an_independent_cuda_runtime_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            probe = {
                "cuda_driver_api_version": "12.8.0",
                "cuda_runtime_version": "12.8.0",
                "diagnostic": None,
                "loaded_library": "/node/lib/libcudart.so.12",
                "query_status": "success",
            }
            environment = dict(CPU_ENVIRONMENT)
            environment.update({
                "GPU_SUITE_GPU_NAME": "node-local-gpu",
                "GPU_SUITE_GPU_UUID": (
                    "GPU-11111111-1111-1111-1111-111111111111"
                ),
                "GPU_SUITE_NVIDIA_DRIVER_VERSION": "575.57.08",
                "GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC": "",
                "GPU_SUITE_NODE_GPU_QUERY_STATUS": "success",
            })
            metadata = build_node_metadata(
                inputs["config"], inputs["manifest"], "run-1", 0, 0,
                "node0", "0" * 64, environment,
                cuda_runtime_probe=lambda: probe,
            )
            self.assertEqual(metadata["cuda_runtime_identity"], probe)
            self.assertEqual(
                metadata["gpu_identity"]["nvidia_driver_version"],
                "575.57.08",
            )
            self.assertIsNone(metadata["gpu_identity"]["diagnostic"])

    def test_gpu_raw_identity_must_match_its_node_metadata(self):
        record = raw_success(implementation="cuda")
        record["gpu_name"] = "node-local-gpu"
        record["gpu_uuid"] = "GPU-11111111-1111-1111-1111-111111111111"
        metadata = {
            "cuda_runtime_identity": {
                "cuda_driver_api_version": record["cuda_driver_version"],
                "cuda_runtime_version": record["cuda_runtime_version"],
                "diagnostic": None,
                "loaded_library": "libcudart.so.12",
                "query_status": "success",
            },
            "gpu_identity": {
                "name": record["gpu_name"],
                "uuid": record["gpu_uuid"],
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.jsonl"
            raw.write_text(dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(
                classify_raw(raw, metadata)["benchmark_status"], "success"
            )
            metadata["gpu_identity"]["uuid"] = (
                "GPU-22222222-2222-2222-2222-222222222222"
            )
            with self.assertRaisesRegex(NodeToolError, "GPU UUID"):
                classify_raw(raw, metadata)
            metadata["gpu_identity"]["uuid"] = record["gpu_uuid"]
            metadata["cuda_runtime_identity"]["cuda_runtime_version"] = "12.7.0"
            with self.assertRaisesRegex(NodeToolError, "CUDA Runtime"):
                classify_raw(raw, metadata)
            metadata["cuda_runtime_identity"]["cuda_runtime_version"] = None
            with self.assertRaisesRegex(NodeToolError, "complete node-matched"):
                classify_raw(raw, metadata)

    def test_all_healthy_nodes_succeed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            mapping = [{"hostname": "node0", "rank": 0}]
            write_wave_metadata(wave_metadata, mapping)
            write_node(nodes, "node0", 0, {
                "benchmark_status": "success",
                "verification_status": "success",
            })
            document, success = collect_results(wave_metadata, nodes)
            self.assertTrue(success)
            self.assertEqual(document["collection_status"], "success")

    def test_one_node_benchmark_failure_does_not_hide_other_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            mapping = [
                {"hostname": "node0", "rank": 0},
                {"hostname": "node1", "rank": 1},
            ]
            write_wave_metadata(wave_metadata, mapping)
            healthy = {
                "benchmark_status": "success",
                "verification_status": "success",
            }
            failed = {
                "benchmark_status": "failure",
                "verification_status": "success",
            }
            raw0 = write_node(nodes, "node0", 0, failed)
            raw1 = write_node(nodes, "node1", 1, healthy)
            before = {raw0: raw0.read_bytes(), raw1: raw1.read_bytes()}
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn("benchmark failure: node0", document["failures"])
            artifact_paths = {item["path"] for item in document["artifacts"]}
            self.assertIn("node0/raw-results.jsonl", artifact_paths)
            self.assertIn("node1/raw-results.jsonl", artifact_paths)
            self.assertEqual(before, {raw0: raw0.read_bytes(), raw1: raw1.read_bytes()})

    def test_missing_node_and_collection_failure_make_job_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            mapping = [
                {"hostname": "node0", "rank": 0},
                {"hostname": "node1", "rank": 1},
            ]
            write_wave_metadata(wave_metadata, mapping)
            healthy = {
                "benchmark_status": "success",
                "verification_status": "success",
            }
            write_node(nodes, "node0", 0, healthy)
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn("missing node directory: node1", document["failures"])

    def test_missing_status_verification_and_collection_failures_are_distinct(self):
        healthy = {
            "benchmark_status": "success",
            "verification_status": "success",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            write_wave_metadata(wave_metadata, [{"hostname": "node0", "rank": 0}])
            node = write_node(nodes, "node0", 0, healthy).parent
            (node / "node-status.json").unlink()
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn("missing node status: node0", document["failures"])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            write_wave_metadata(wave_metadata, [{"hostname": "node0", "rank": 0}])
            write_node(nodes, "node0", 0, {
                "benchmark_status": "success",
                "verification_status": "failure",
            })
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn("verification failure: node0", document["failures"])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            write_wave_metadata(wave_metadata, [{"hostname": "node0", "rank": 0}])
            node = write_node(nodes, "node0", 0, healthy).parent
            status_path = node / "node-status.json"
            status = load(status_path)
            status["collection_status"] = "failure"
            status["fatal_infrastructure_failure"] = True
            status["status"] = "failure"
            status_path.write_bytes(dump_bytes(status))
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn("artifact collection failure: node0", document["failures"])


class NodeRunnerIsolationTests(unittest.TestCase):
    def test_signal_status_is_recoverable_but_records_failure(self):
        status = build_node_status(
            "run-1", 0, 0, "node0", 143, 143,
            "success", "success", "success", "warning", None,
            {"telemetry_status": "unavailable"}, ["terminated"], "TERM",
        )
        self.assertEqual(status["termination_signal"], "TERM")
        self.assertEqual(status["benchmark_status"], "failure")
        self.assertFalse(status["fatal_infrastructure_failure"])

    def test_benchmark_failure_collects_status_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            run_root = directory / "results" / "run-node-test"
            nodes = run_root / "waves" / "0" / "nodes"
            nodes.mkdir(parents=True)
            scratch = directory / "scratch"
            scratch.mkdir()
            fake_runner = directory / "fake_run_suite.py"
            fake_runner.write_text(
                "import argparse, sys\n"
                "from pathlib import Path\n"
                "from gpu_suite.config import load_config\n"
                "from gpu_suite.ordering import implementation_order\n"
                "from gpu_suite.runner import build_schedule, synthetic_result\n"
                "from gpu_suite.strict_json import dumps\n"
                "p=argparse.ArgumentParser(add_help=False)\n"
                "p.add_argument('--config', type=Path); p.add_argument('--output', type=Path)\n"
                "p.add_argument('--run-id'); p.add_argument('--system-label'); p.add_argument('--wave', type=int)\n"
                "p.add_argument('--node-index', type=int); p.add_argument('--hostname')\n"
                "p.add_argument('--runtime-environment-sha256')\n"
                "a,_=p.parse_known_args()\n"
                "c=load_config(a.config)\n"
                "x={'config':c,'cpu_threads':c['cpu_threads'],'device':0,'hostname':a.hostname,"
                "'implementation_order':implementation_order(a.node_index,a.wave),'node_index':a.node_index,"
                "'run_id':a.run_id,'scheduler':None,'scheduler_job_id':None,'system_label':a.system_label,'wave':a.wave}\n"
                "item=build_schedule(c,{'entries':[],'manifest_schema_version':1},x)[0]\n"
                "r=synthetic_result(item,x,'0'*64,a.runtime_environment_sha256,{},0,True,'benchmark','fixture failure',1,None,None)\n"
                "a.output.write_text(dumps(r)+'\\n',encoding='utf-8')\n"
                "sys.exit(1)\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(CPU_ENVIRONMENT)
            environment["OMPI_COMM_WORLD_RANK"] = "0"
            environment["PYTHONPATH"] = str(ROOT / "tools")
            command = [
                "bash", str(PEGASUS_DIRECTORY / "run_node.sh"),
                "--repository-root", str(ROOT),
                "--run-root", str(run_root),
                "--config", str(inputs["config"]),
                "--manifest", str(inputs["manifest"]),
                "--build-metadata", str(inputs["metadata"]),
                "--run-id", "run-node-test",
                "--wave", "0",
                "--system-label", "local-node",
                "--scheduler", "test",
                "--scheduler-job-id", "fixture.1",
                "--runtime-environment-sha256", "0" * 64,
                "--scratch-root", str(scratch),
                "--run-suite-script", str(fake_runner),
            ]
            completed = subprocess.run(
                command, env=environment, text=True, capture_output=True,
                check=False, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            hostname = socket.gethostname()
            status_path = nodes / hostname / "node-status.json"
            status = load(status_path)
            self.assertEqual(status["benchmark_status"], "failure")
            self.assertEqual(status["collection_status"], "success")
            self.assertFalse(status["fatal_infrastructure_failure"])
            self.assertTrue((nodes / hostname / "raw-results.jsonl").is_file())

            second = list(command)
            second[second.index("fixture.1")] = "fixture.2"
            collided = subprocess.run(
                second, env=environment, text=True, capture_output=True,
                check=False, timeout=30,
            )
            self.assertNotEqual(collided.returncode, 0)


if __name__ == "__main__":
    unittest.main()
