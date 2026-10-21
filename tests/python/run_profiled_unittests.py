#!/usr/bin/env python3
"""Run the profile-independent or CPU-fixture Python unittest subset."""

import argparse
import ast
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path


CPU_FIXTURE_MODULE_PREFIXES = (
    "test_cufft_fixture_runtime.",
    "test_rocfft_fixture_runtime.",
    "test_overflow_runtime.",
    "test_phase3_fixture_runtime.",
)
CPU_FIXTURE_TEST_IDS = frozenset({
    "test_runner.RunnerTests."
    "test_real_fixture_runs_through_single_writer_and_validates",
})



# Audited IDs, not counts. Update a digest only after reviewing every new/removed
# case. Checked before unittest discovery imports any test modules.
AUDITED_MODULE_IDS = {'test_sirius_amd_prepare': 'ac683d636a4ed5c4cc2869c2405869c35dbc32c35d76ac3a807d0562c808dea5',
 'test_sirius_amd_production': '25f385db672ea9029bbab606a59157d99da8870bc9fe96222ba04746a384507b',
 'test_sirius_amd_smoke': '13f91475ce925b866ddf4739eb4b2610d04629fd20e6f00ab849fae4aa9c23e5',
 'test_sirius_amd_smoke_campaign': '760c2e71e6fa834acda03826cc9c0a23bff15d0ea7dbbd908e63415e82ef4c8b',
 'test_sirius_amd_smoke_evidence': 'aacef55694427cf7102c35348091e8786e82aa28347c5fe922b9dee711cec142',
 'test_sirius_amd_smoke_job': 'c84d9cbfb58685ca691023abab1ece3ff5de476543d557dbd4aa8d5b4517b978',
 'test_sirius_amd_smoke_source': '46f16b03b845cb3599fe37faadd6eb8f061a2838b54487525794b97318c178e0',
 'test_amd_blas_sparse': '37174c27361c2216335373c9d34fb134dc9c5ebfe5e137460c13fd5d04c8b6e2',
 'test_amd_cmake': '0c3b2dc3c9409def48c22cbd9da62fc8a6a16a0b75a399bb9f90b901640f6b8e',
 'test_amd_cpu_posix': 'efea84f82227f825f1e8c40283455e86987d36049c207c63cbe5d845074bf7d6',
 'test_amd_families': 'be15924f6659400e561fc874b5322f00555c3cad62a32cc6766bca11b154d7cb',
 'test_amd_gpu_build_probes': '21ccc628c5e90eb5850d38a96b9100529f7bb9aa662c250c540310c4535a6f05',
 'test_amd_fortran_build_fixes': '621a281879f742b8ba74ca68dd095389763eaaadd32119a29c613a3762ef098e',
 'test_amd_metadata': '810bf7f1d10c16289a78cdd7fb818127e1f0a180bd8d624acacebc9a1b018c26',
 'test_amd_rand_thrust': '263cdd7b39648d622294b8c1296c6cc6a7273684e5473feeaa7e5ddc56290de7',
 'test_aocl_sparse': '6fbf14b3eeb3fadfba9926cc5bd5366f6d04c5e3ce3b528f5be4fab37f1d7390',
 'test_aocl_sparse_tools': 'fa122dd3d6c523d2fca8333000ddcb1a36fff49c7c987cb9f053ebeacb68b988',
 'test_amd_solver': 'c9b50d4b8d6ec12d6452b1fda4017e2b5da4e73e44c6f99671c51f6aa17fa8dc',
 'test_rocfft_module_build': '218fe1fe4d5b6bea3269fc8284d3e094c91bd1078f59eec240a54258b60ac547',
 'test_rocfft_build_only': 'a26cf1f6d508e98c709d2f2bd5ae98a179e56975a84ec76141ce4d2a372b113e',
 'test_fftw_metadata': '58e19ef4359bf9f8314aea82b3c0e98361e4476e129224cc86e1cd41c724e6cd',
 'test_aggregation': 'ae78f974d5804b334f74e0da2aabfef4ec2bd203911791cf94dea4914c34d9c2',
 'test_build_metadata': '1dd4c64bb6bf77490dab2da378d34bb2282950105bac8e8faf922154c529571b',
 'test_config': '001fbc7b3ff85d173d3fa2c3c2d163d34004f1a90a33eb14f0399f6a80f5d637',
 'test_cufft_fixture_runtime': '72235f44e22d52349e5aa1acc5a7604b8b3fbdc8544d5e21f68fd6ac271b8785',
 'test_cufft_source_policy': 'e12813c47057b52e6444c2f678caebb4108fb31cf2d90ae79530a18264216e25',
 'test_documentation': '6c00a881f2a2ac72bc130ad31fb973278a85191e661b28e76dce79693166ffbd',
 'test_fortran': 'fefa72185a86327ce4dee38ef49852c59f7d42f35c9d77f98755655b954233c8',
 'test_generic_cpu_providers': 'c143707bb5c72190deea60dbc2338c7a5e7f272ae444502286715af2de8107c3',
 'test_hashing': 'c4ad0e6d084ef6490ed564e9723c8fd71fa9e8efedd8d458d2fe077d61095cd1',
 'test_hip_metadata': 'f5f350cdc2bf034a5d7f5d74df659818d79c4250e54bc886603f3e8ba3ab9488',
 'test_local_metadata': '8ade8ee74922e31766c872cfd5cd1cf625c52faa50ff7d66c0f3c6c063f158fb',
 'test_manifest': '1d5469e5cf676b28df5cee0449854c8d3450aa31aa849317c9684bb976e44120',
 'test_non_git_build': 'ad62161f27d7c6c0475f31f564f853d7671f5bb6a5913d455174ac1bd85213f0',
 'test_ordering': '19c965c443f4477569abe31f1794cdfae27bd26cb2445622e7412b62ce3f2f42',
 'test_overflow_runtime': '29244fc4c38a24118c9b85ba3ebc9daf478f24a0decc1fd22df6dcf8d218e2d4',
 'test_pegasus_node_collection': 'e98bee5a7539bff060f02475fb744fbf8dc9a00cc2b1f7cc9897378933eab773',
 'test_pegasus_prepare_render': 'ef71f354b896730cede5adfbc9511899fd6d7edfa1a0d565321a983ec3a7c01a',
 'test_pegasus_runtime_telemetry': '01cc4a47bac3e57778aa85237fc8f397a4a591f46a8b8ba4ec1bc0347a982476',
 'test_phase3_fixture_runtime': '73355db30206a6873928afa40e4489c683aa0fff1f790e029f804fcd85d81084',
 'test_phase3_source_policy': '9df5b50fb9459aa1b6b5b0bc523f9ab402bd4fb3eb940995a1a69e4411322517',
 'test_plotting': '5afe638dc6345651b7dae5999edcc770cf3a31dfc5757a958bce07783a5106f2',
 'test_python39_check': '4542641401d6545dd32d99c18de79d1103376e6274bc3fb1d2597e0a3921bddd',
 'test_results_io': '2045cdb650ee656916ab434b3bd4da5197408913fa6e8742ec28db42d15095fe',
 'test_review_source_policy': '67dec7588a0e302d53d9820073a1860760f90253688763c4449f469baf914e76',
 'test_rocfft_cmake': '9b949967c9368b7b1730f9f4d8a8ac7e20cb78bef58d4f7672d5ed33400b759e',
 'test_rocfft_fixture_runtime': '55f59551d40542c71e7a6cf848a732b70d5fc313941008eeaedf41ac0a0c2e9b',
 'test_rocfft_smoke': '3bf8c9d12f7c7ba1f252fe2e379f1a5bef4500d272b9cf05d1d1ef3a02f840c5',
 'test_rocfft_sources': '39e02fcd6142e0bc71f6d585cdfad57e33d03dcaef6e4caba09b6bf05a3be213',
 'test_rocfft_compute': 'a5de9bccb979c30dbca13be7c18e5987b6db4dc33e3eed9d025992f06f17b10e',
 'test_rocfft_job': '077a18a8617fb6c58aa3056810c2c4691af4a63bd29ec1bf247a8494ef668360',
 'test_rocfft_probe_compile': 'ad5931e1f76b30ec4ec1e8d6061e50504bc5234f6e6eca88767779be563ad538',
 'test_rocfft_tools': '215ebd4f191c7b388fbc55f98132b7d73e50c704b74103f4b917c3b1ed7ece1d',
 'test_runner': '6a9df02d5de3bc025f76b19fbdd0f4fba3d34bc1be3b0c15b3c8aae7c39aefc5',
 'test_safe_local_validation': 'c6644d32805a789e599fc7570d64e90f9e0f87f5176d469a9ca518b13fb945a7',
 'test_scheduler_identity': '11d27f0f535e34c0e72ae9533cce26d77ae0ee8f076c9d4e22fef0d14b03370f',
 'test_schema': '712a1e552fb3055780061f8588a6d1009df5fa577ac4efcb087bfa2403f3d2f6',
 'test_sirius_remote': 'a7aab7a08b9ba4f532cc6d10645db6b3add700b6e6c1993aef3caa800f5e09bf',
 'test_sirius_amd_build': 'b7c0f56ddfff8dc10a75d2d1557d62dfdf8c0b7e879e5b7cef126b2b658eca9b',
 'test_sirius_fftw_probe': '2ebe5e9c722e9a175fde27a26ba8c468eb8e82eb68532fce6606b4951dcab51d',
 'test_sirius_fftw_modules': 'a472cfc8ccf0fcdb34e094c62c399abf478b6c998d7036688c15af3229e022d7',
 'test_sirius_fftw_threads': 'cd0154146969d641603abf14eda75f32d8cb33bd7290807f8dc6263c89cab3d7',
 'test_strict_json': '7ba5c3fffa8f502da31761429132486b034c364984a780f5ef8ecb4b30045677',
 'test_validation': '4c2fa22a056372a29c2e4bde6bd3dd66410e07d8531e87b00af260765a0526ed'}

POLICY_EXCLUSIONS = {
    "test_pegasus_runtime_telemetry.RuntimeEnvironmentTests."
    "test_complete_document_and_stable_normalized_ldd":
        "creates a symbolic link; local policy prohibits symlink creation",
    "test_pegasus_node_collection.NodeRunnerIsolationTests."
    "test_benchmark_failure_collects_status_and_exits_zero":
        "starts real run_node.sh, including machine probes",
}


def audit_inventory(start_directory):
    found = {}
    all_ids = set()
    for path in sorted(Path(start_directory).glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"))
        names = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names.extend(
                    path.stem + "." + node.name + "." + method.name
                    for method in node.body
                    if isinstance(method, ast.FunctionDef)
                    and method.name.startswith("test_")
                )
        if len(names) != len(set(names)):
            raise ValueError("duplicate test IDs in " + path.name)
        all_ids.update(names)
        found[path.stem] = hashlib.sha256(
            "\n".join(sorted(names)).encode("utf-8")
        ).hexdigest()
    if found != AUDITED_MODULE_IDS:
        changed = sorted(name for name in set(found) | set(AUDITED_MODULE_IDS)
                         if found.get(name) != AUDITED_MODULE_IDS.get(name))
        raise ValueError("unaudited or missing test inventory: " + ", ".join(changed))
    if not set(POLICY_EXCLUSIONS).issubset(all_ids):
        raise ValueError("policy-excluded test disappeared; review inventory")
    return all_ids


class RecordedResult(unittest.TextTestResult):
    """One explicit outcome per test, including failed subtests and fixtures."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {}

    def record(self, test, status, reason):
        previous = self.records.get(test.id())
        if previous and previous["status"] == "FAIL":
            return
        self.records[test.id()] = {
            "id": test.id(), "status": status, "reason": reason,
        }

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, "PASS", "unittest completed successfully")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, "FAIL", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, "SKIP", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.record(test, "SKIP", "unittest expected failure")

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.record(test, "FAIL", "unittest unexpected success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.record(test, "FAIL", self._exc_info_to_string(err, test))


def complete_records(selected_ids, result):
    records = dict(result.records)
    for test_id in selected_ids:
        if test_id in records:
            continue
        fixture_skip = next((row for key, row in records.items()
                             if row["status"] == "SKIP"
                             and (key == "setUpClass (" + test_id.rsplit(".", 1)[0] + ")"
                                  or key == "setUpModule (" + test_id.split(".", 1)[0] + ")")), None)
        if fixture_skip is not None:
            records[test_id] = {"id": test_id, "status": "SKIP", "reason": fixture_skip["reason"]}
        else:
            records[test_id] = {
                "id": test_id, "status": "FAIL",
                "reason": "selected test has no recorded completion evidence",
            }
    return list(records.values())


def iter_test_cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_test_cases(item)
        else:
            yield item


def requires_cpu_fixture(test_case):
    test_id = test_case.id()
    return (
        test_id in CPU_FIXTURE_TEST_IDS
        or test_id.startswith(CPU_FIXTURE_MODULE_PREFIXES)
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-directory", required=True, type=Path)
    parser.add_argument(
        "--profile", required=True,
        choices=("independent", "cpu-fixture", "safe-independent")
    )
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args(argv)
    start_directory = arguments.start_directory.resolve()
    safe = arguments.profile == "safe-independent"
    if safe or os.environ.get("GPU_SUITE_SAFE_LOCAL") == "1":
        expected_ids = audit_inventory(start_directory)
    else:
        expected_ids = None
    sys.path.insert(0, str(start_directory))
    loader = unittest.TestLoader()
    discovered = loader.discover(str(start_directory), pattern="test_*.py")
    if loader.errors:
        for error in loader.errors:
            print(error, file=sys.stderr)
        return 1
    all_cases = list(iter_test_cases(discovered))
    if expected_ids is not None and {case.id() for case in all_cases} != expected_ids:
        raise ValueError("discovered test IDs differ from audited source IDs")
    cpu_cases = [case for case in all_cases if requires_cpu_fixture(case)]
    if arguments.profile == "cpu-fixture":
        selected = cpu_cases
    else:
        selected = [case for case in all_cases if not requires_cpu_fixture(case)]
    excluded = []
    if safe:
        excluded = [{"id": case.id(), "status": "POLICY_EXCLUDED",
                     "reason": POLICY_EXCLUSIONS[case.id()]}
                    for case in selected if case.id() in POLICY_EXCLUSIONS]
        selected = [case for case in selected if case.id() not in POLICY_EXCLUSIONS]
    if not selected:
        print("profile selected no Python tests", file=sys.stderr)
        return 1
    selected_ids = [case.id() for case in selected]
    print("Python unittest profile {0}: {1} selected, {2} CPU-fixture, "
          "{3} total".format(arguments.profile, len(selected), len(cpu_cases), len(all_cases)))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(
        unittest.TestSuite(selected)
    )
    records = complete_records(selected_ids, result)
    if safe:
        for prefix in ("test_sirius_remote.", "test_safe_local_validation."):
            required = [row for row in records if row["id"].startswith(prefix)]
            if not required or any(row["status"] != "PASS" for row in required):
                records.append({"id": prefix + "required-coverage", "status": "FAIL",
                                "reason": "all new local safety tests must execute and pass"})
    report = arguments.report
    if report is None and arguments.profile == "cpu-fixture":
        report_name = os.environ.get("GPU_SUITE_SAFE_LOCAL_CPU_REPORT")
        report = Path(report_name) if report_name else None
    if report is not None:
        document = {"profile": arguments.profile, "selected": selected_ids,
                    "inventory": [case.id() for case in all_cases],
                    "results": records + excluded}
        with report.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return 0 if result.wasSuccessful() and not any(
        row["status"] == "FAIL" for row in records
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
