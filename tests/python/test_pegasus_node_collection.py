import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from gpu_suite.hashing import sha256_file
from gpu_suite.results_io import load_raw_results
from gpu_suite.strict_json import dump_bytes, dumps, load

from pegasus_support import CPU_ENVIRONMENT, write_campaign_inputs
from support import ROOT, raw_success


PEGASUS_DIRECTORY = ROOT / "jobs" / "pegasus"
SCHEDULER = "NQSV"
RAW_SCHEDULER_JOB_ID = "0:866211.nqsv"
sys.path.insert(0, str(PEGASUS_DIRECTORY))

from collect_results import collect_results  # noqa: E402
from node_tools import (  # noqa: E402
    NodeToolError,
    build_node_metadata,
    build_node_status,
    classify_raw,
)


def write_wave_metadata(path, mapping, **extra):
    document = {
        "rank_host_mapping": mapping,
        "run_id": "run-1",
        "scheduler": SCHEDULER,
        "scheduler_job_id": RAW_SCHEDULER_JOB_ID,
        "wave": 0,
        "wave_metadata_schema_version": 1,
    }
    document.update(extra)
    path.write_bytes(dump_bytes(document))


def write_node(nodes, hostname, rank, classification):
    node = nodes / hostname
    node.mkdir()
    raw = raw_success(
        node_index=rank, hostname=hostname, scheduler=SCHEDULER,
        scheduler_job_id=RAW_SCHEDULER_JOB_ID,
    )
    (node / "raw-results.jsonl").write_text(dumps(raw) + "\n", encoding="utf-8")
    (node / "node-metadata.json").write_bytes(dump_bytes({
        "block_id": "run-1|0|{0}".format(hostname),
        "hostname": hostname,
        "node_index": rank,
        "node_metadata_schema_version": 1,
        "run_id": "run-1",
        "scheduler": SCHEDULER,
        "scheduler_job_id": RAW_SCHEDULER_JOB_ID,
        "wave": 0,
    }))
    (node / "logs").mkdir()
    (node / "logs" / "runner.log").write_text("preserved\n", encoding="utf-8")
    status = build_node_status(
        "run-1", 0, rank, hostname,
        0 if classification["benchmark_status"] == "success" else 1,
        0 if classification["benchmark_status"] == "success" else 1,
        "success", "success", "success", "success", classification,
        {"telemetry_status": "success"}, [], None,
        scheduler=SCHEDULER,
        scheduler_job_id=RAW_SCHEDULER_JOB_ID,
    )
    (node / "node-status.json").write_bytes(dump_bytes(status))
    return node / "raw-results.jsonl"


class CollectionTests(unittest.TestCase):
    def test_curand_metadata_selects_cpu_engine_from_language_and_backend(self):
        cases = (
            (None, "pilot.json", "std::mt19937_64"),
            ("c-cpp", "pilot.json", "std::mt19937_64"),
            ("fortran", "fortran/pilot.json", "Fortran random_number"),
        )
        for language, config_name, engine in cases:
            with self.subTest(language=language), tempfile.TemporaryDirectory() as temporary:
                config = load(ROOT / "configs" / config_name)
                if language is None:
                    config.pop("source_language", None)
                else:
                    config["source_language"] = language
                inputs = write_campaign_inputs(Path(temporary), config=config)
                metadata = build_node_metadata(
                    inputs["config"], inputs["manifest"], "run-1", 0, 0,
                    "node0", "0" * 64, dict(CPU_ENVIRONMENT),
                    cuda_runtime_probe=lambda: {
                        "cuda_driver_api_version": None,
                        "cuda_runtime_version": None,
                        "diagnostic": "CPU-only fixture",
                        "loaded_library": None,
                        "query_status": "unavailable",
                    },
                )
                self.assertEqual(metadata["curand"]["cpu_engine"], engine)
                self.assertEqual(
                    metadata["curand"]["cuda_generator"], "pseudo-default"
                )

    def test_curand_metadata_rejects_language_backend_mismatch(self):
        for config_name, language in (
            ("pilot.json", "fortran"),
            ("fortran/pilot.json", "c-cpp"),
        ):
            with self.subTest(language=language), tempfile.TemporaryDirectory() as temporary:
                config = load(ROOT / "configs" / config_name)
                config["source_language"] = language
                inputs = write_campaign_inputs(Path(temporary), config=config)
                with self.assertRaisesRegex(NodeToolError, "CPU backend differs"):
                    build_node_metadata(
                        inputs["config"], inputs["manifest"], "run-1", 0, 0,
                        "node0", "0" * 64, dict(CPU_ENVIRONMENT),
                        cuda_runtime_probe=lambda: {
                            "cuda_driver_api_version": None,
                            "cuda_runtime_version": None,
                            "diagnostic": "CPU-only fixture",
                            "loaded_library": None,
                            "query_status": "unavailable",
                        },
                    )

    @staticmethod
    def _curand_node_metadata(record, engine):
        return {
            "curand": {
                "cpu_engine": engine,
                "cuda_generator": "pseudo-default",
            },
            "cuda_runtime_identity": {
                "cuda_driver_api_version": record["cuda_driver_version"],
                "cuda_runtime_version": record["cuda_runtime_version"],
            },
            "gpu_identity": {
                "name": record["gpu_name"],
                "uuid": record["gpu_uuid"],
            },
        }

    def test_curand_cpu_raw_engine_matches_node_metadata(self):
        for engine, backend in (
            ("std::mt19937_64", "cpu-std-random-serial"),
            ("Fortran random_number", "cpu-fortran-random-serial"),
        ):
            with self.subTest(engine=engine), tempfile.TemporaryDirectory() as temporary:
                record = raw_success(benchmark="curand")
                record["parameters"]["cpu_engine"] = engine
                record["cpu_backend"] = backend
                record["library_name"] = backend
                if backend == "cpu-fortran-random-serial":
                    record["parameters"]["source_language"] = "fortran"
                raw = Path(temporary) / "raw.jsonl"
                raw.write_text(dumps(record) + "\n", encoding="utf-8")
                metadata = self._curand_node_metadata(record, engine)
                classification = classify_raw(raw, metadata)
                self.assertEqual(classification["benchmark_status"], "success")
                self.assertEqual(classification["verification_status"], "success")
                self.assertEqual(classification["record_count"], 1)

    def test_fortran_curand_raw_rejects_cpp_node_engine(self):
        record = raw_success(benchmark="curand")
        record["parameters"].update({
            "cpu_engine": "Fortran random_number",
            "source_language": "fortran",
        })
        record["cpu_backend"] = "cpu-fortran-random-serial"
        record["library_name"] = "cpu-fortran-random-serial"
        metadata = self._curand_node_metadata(record, "std::mt19937_64")
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw.jsonl"
            raw.write_text(dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(NodeToolError, "cuRAND CPU engine differs"):
                classify_raw(raw, metadata)

    def test_successful_curand_cpu_raw_requires_both_engine_records(self):
        for missing in ("raw", "node", "curand"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                record = raw_success(benchmark="curand")
                metadata = self._curand_node_metadata(record, "std::mt19937_64")
                if missing == "raw":
                    del record["parameters"]["cpu_engine"]
                elif missing == "node":
                    del metadata["curand"]["cpu_engine"]
                else:
                    del metadata["curand"]
                raw = Path(temporary) / "raw.jsonl"
                raw.write_text(dumps(record) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(NodeToolError, "node-matched cuRAND CPU engine"):
                    classify_raw(raw, metadata)

    def test_curand_gpu_generator_is_not_compared_with_cpu_engine(self):
        for implementation in ("cuda", "openacc"):
            with self.subTest(implementation=implementation), tempfile.TemporaryDirectory() as temporary:
                record = raw_success(benchmark="curand", implementation=implementation)
                self.assertNotIn("cpu_engine", record["parameters"])
                self.assertEqual(
                    record["parameters"]["generator_algorithm"],
                    "CURAND_RNG_PSEUDO_DEFAULT",
                )
                metadata = self._curand_node_metadata(record, "Fortran random_number")
                raw = Path(temporary) / "raw.jsonl"
                raw.write_text(dumps(record) + "\n", encoding="utf-8")
                self.assertEqual(classify_raw(raw, metadata)["benchmark_status"], "success")

    def test_curand_missing_engine_preserves_failed_and_skipped_status(self):
        for status in ("failure", "skipped"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                record = raw_success(benchmark="curand")
                del record["parameters"]["cpu_engine"]
                record.update({
                    "status": status,
                    "attempted": status == "failure",
                    "failure_origin": "benchmark" if status == "failure" else "prior-failure",
                    "exit_code": 1 if status == "failure" else None,
                    "message": "fixture execution failure",
                    "verification_status": "skipped",
                    "measurement_start_timestamp": None,
                    "measurement_end_timestamp": None,
                    "elapsed_total_sec": None,
                    "elapsed_sec": None,
                })
                metadata = self._curand_node_metadata(record, "Fortran random_number")
                raw = Path(temporary) / "raw.jsonl"
                raw.write_text(dumps(record) + "\n", encoding="utf-8")
                classification = classify_raw(raw, metadata)
                self.assertEqual(classification["benchmark_status"], "failure")
                self.assertEqual(classification["status_counts"], {status: 1})

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
                scheduler=SCHEDULER,
                scheduler_job_id=RAW_SCHEDULER_JOB_ID,
            )
            self.assertEqual(metadata["cuda_runtime_identity"], probe)
            self.assertEqual(
                metadata["gpu_identity"]["nvidia_driver_version"],
                "575.57.08",
            )
            self.assertIsNone(metadata["gpu_identity"]["diagnostic"])
            self.assertEqual(
                metadata["scheduler_job_id"], RAW_SCHEDULER_JOB_ID
            )

    def test_node_specific_provenance_survives_across_waves(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = write_campaign_inputs(directory)
            runtime_sha256 = "a" * 64
            probe = {
                "cuda_driver_api_version": "13.0.0",
                "cuda_runtime_version": "13.0.96",
                "diagnostic": None,
                "loaded_library": "/node/lib/libcudart.so.13",
                "query_status": "success",
            }

            def metadata(wave, hostname, uuid, scheduler_job_id):
                environment = dict(CPU_ENVIRONMENT)
                environment.update({
                    "GPU_SUITE_GPU_NAME": "NVIDIA H100 80GB HBM3",
                    "GPU_SUITE_GPU_UUID": uuid,
                    "GPU_SUITE_NVIDIA_DRIVER_VERSION": "580.95.05",
                    "GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC": "",
                    "GPU_SUITE_NODE_GPU_QUERY_STATUS": "success",
                })
                return build_node_metadata(
                    inputs["config"], inputs["manifest"], "run-1", wave,
                    0, hostname, runtime_sha256, environment,
                    cuda_runtime_probe=lambda: probe,
                    scheduler=SCHEDULER,
                    scheduler_job_id=scheduler_job_id,
                )

            first = metadata(
                0, "bnode111",
                "GPU-11111111-1111-1111-1111-111111111111",
                "0:866328.nqsv",
            )
            second = metadata(
                1, "bnode117",
                "GPU-22222222-2222-2222-2222-222222222222",
                "0:866329.nqsv",
            )
            self.assertEqual(
                first["runtime_environment_sha256"],
                second["runtime_environment_sha256"],
            )
            self.assertEqual(first["hostname"], "bnode111")
            self.assertEqual(second["hostname"], "bnode117")
            self.assertNotEqual(
                first["gpu_identity"]["uuid"],
                second["gpu_identity"]["uuid"],
            )
            self.assertEqual(first["scheduler_job_id"], "0:866328.nqsv")
            self.assertEqual(second["scheduler_job_id"], "0:866329.nqsv")
            self.assertEqual((first["wave"], second["wave"]), (0, 1))

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

    def test_raw_scheduler_identity_must_match_node_metadata(self):
        metadata = {
            "cuda_runtime_identity": {
                "cuda_driver_api_version": None,
                "cuda_runtime_version": None,
            },
            "gpu_identity": {"name": None, "uuid": None},
            "scheduler": SCHEDULER,
            "scheduler_job_id": RAW_SCHEDULER_JOB_ID,
        }
        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / "raw.jsonl"
            record = raw_success(
                scheduler=SCHEDULER,
                scheduler_job_id=RAW_SCHEDULER_JOB_ID,
            )
            raw_path.write_text(dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(
                classify_raw(raw_path, metadata)["benchmark_status"],
                "success",
            )
            metadata["scheduler_job_id"] = "0:866212.nqsv"
            with self.assertRaisesRegex(NodeToolError, "scheduler identity"):
                classify_raw(raw_path, metadata)

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

    def test_collector_validates_linked_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            job_master = directory / "job-master"
            job_master.mkdir()
            runtime_sha256 = "a" * 64
            manifest_sha256 = "b" * 64
            evidence_path = (
                job_master / "runtime-environment-evidence.json"
            )
            evidence = {
                "executables_manifest_sha256": manifest_sha256,
                "runtime_environment_evidence_schema_version": 1,
                "runtime_environment_sha256": runtime_sha256,
            }
            evidence_path.write_bytes(dump_bytes(evidence))
            wave_metadata = directory / "wave-metadata.json"
            write_wave_metadata(
                wave_metadata, [{"hostname": "node0", "rank": 0}],
                executables_manifest_sha256=manifest_sha256,
                runtime_environment_evidence_sha256=sha256_file(evidence_path),
                runtime_environment_sha256=runtime_sha256,
            )
            write_node(nodes, "node0", 0, {
                "benchmark_status": "success",
                "verification_status": "success",
            })
            document, success = collect_results(wave_metadata, nodes)
            self.assertTrue(success, document["failures"])

            evidence["probe_output"] = "changed"
            evidence_path.write_bytes(dump_bytes(evidence))
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn(
                "runtime environment evidence SHA-256 mismatch",
                document["failures"],
            )

            evidence["runtime_environment_sha256"] = "c" * 64
            evidence_path.write_bytes(dump_bytes(evidence))
            wave = load(wave_metadata)
            wave["runtime_environment_evidence_sha256"] = sha256_file(
                evidence_path
            )
            wave_metadata.write_bytes(dump_bytes(wave))
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn(
                "runtime environment evidence identity mismatch",
                document["failures"],
            )

            evidence["runtime_environment_sha256"] = runtime_sha256
            evidence["executables_manifest_sha256"] = "d" * 64
            evidence_path.write_bytes(dump_bytes(evidence))
            wave["runtime_environment_evidence_sha256"] = sha256_file(
                evidence_path
            )
            wave_metadata.write_bytes(dump_bytes(wave))
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertIn(
                "runtime environment evidence manifest mismatch",
                document["failures"],
            )

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

    def test_collector_rejects_node_and_raw_scheduler_identity_mismatch(self):
        healthy = {
            "benchmark_status": "success",
            "verification_status": "success",
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            write_wave_metadata(
                wave_metadata, [{"hostname": "node0", "rank": 0}]
            )
            node = write_node(nodes, "node0", 0, healthy).parent
            status_path = node / "node-status.json"
            status = load(status_path)
            status["scheduler_job_id"] = "0:866212.nqsv"
            status_path.write_bytes(dump_bytes(status))
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertTrue(any(
                "node status mismatch" in failure
                for failure in document["failures"]
            ))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            nodes = directory / "nodes"
            nodes.mkdir()
            wave_metadata = directory / "wave-metadata.json"
            write_wave_metadata(
                wave_metadata, [{"hostname": "node0", "rank": 0}]
            )
            raw_path = write_node(nodes, "node0", 0, healthy)
            record = raw_success(
                scheduler=SCHEDULER,
                scheduler_job_id="0:866212.nqsv",
            )
            raw_path.write_text(dumps(record) + "\n", encoding="utf-8")
            document, success = collect_results(wave_metadata, nodes)
            self.assertFalse(success)
            self.assertTrue(any(
                "raw scheduler identity" in failure
                for failure in document["failures"]
            ))


class NodeRunnerIsolationTests(unittest.TestCase):
    def test_signal_status_is_recoverable_but_records_failure(self):
        status = build_node_status(
            "run-1", 0, 0, "node0", 143, 143,
            "success", "success", "success", "warning", None,
            {"telemetry_status": "unavailable"}, ["terminated"], "TERM",
            scheduler=SCHEDULER,
            scheduler_job_id=RAW_SCHEDULER_JOB_ID,
        )
        self.assertEqual(status["termination_signal"], "TERM")
        self.assertEqual(status["benchmark_status"], "failure")
        self.assertFalse(status["fatal_infrastructure_failure"])
        self.assertEqual(status["scheduler_job_id"], RAW_SCHEDULER_JOB_ID)

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
                "p.add_argument('--scheduler'); p.add_argument('--scheduler-job-id')\n"
                "a,_=p.parse_known_args()\n"
                "c=load_config(a.config)\n"
                "x={'config':c,'cpu_threads':c['cpu_threads'],'device':0,'hostname':a.hostname,"
                "'implementation_order':implementation_order(a.node_index,a.wave),'node_index':a.node_index,"
                "'run_id':a.run_id,'scheduler':a.scheduler,'scheduler_job_id':a.scheduler_job_id,'system_label':a.system_label,'wave':a.wave}\n"
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
                "--scheduler", SCHEDULER,
                "--scheduler-job-id", RAW_SCHEDULER_JOB_ID,
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
            self.assertEqual(status["scheduler_job_id"], RAW_SCHEDULER_JOB_ID)
            metadata = load(nodes / hostname / "node-metadata.json")
            self.assertEqual(
                metadata["scheduler_job_id"], RAW_SCHEDULER_JOB_ID
            )
            raw_path = nodes / hostname / "raw-results.jsonl"
            self.assertTrue(raw_path.is_file())
            self.assertEqual(
                load_raw_results(raw_path)[0]["scheduler_job_id"],
                RAW_SCHEDULER_JOB_ID,
            )
            self.assertTrue(any(
                path.name.startswith("gpu-library-suite-866211-")
                for path in scratch.iterdir()
            ))

            second = list(command)
            second[second.index(RAW_SCHEDULER_JOB_ID)] = "0:866212.nqsv"
            collided = subprocess.run(
                second, env=environment, text=True, capture_output=True,
                check=False, timeout=30,
            )
            self.assertNotEqual(collided.returncode, 0)


if __name__ == "__main__":
    unittest.main()
