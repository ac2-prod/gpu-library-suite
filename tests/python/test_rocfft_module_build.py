"""Exercise module/profile orchestration with fake tools; no SSH or compilation."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "jobs/sirius/rocfft_module_build.sh"


class RocfftModuleBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rocfft-modules-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        self.run = self.source / "runs/test-build"
        self.run.mkdir(parents=True)
        self.rocm = self.base / "rocm"
        self.rocm.mkdir()
        self.trace = self.base / "trace.jsonl"
        self.modules = self.base / "modules.txt"
        for directory in ("base-bin", "gcc-bin", "amd-bin"):
            (self.base / directory).mkdir()
        self.gcc_libraries = [self.base / "toolset/usr/lib", self.base / "toolset/usr/lib64"]
        self.aocl = self.base / "aocl"
        self.unrelated = self.base / "unrelated"
        for directory in self.gcc_libraries + [self.aocl / "lib", self.aocl / "include", self.unrelated]:
            directory.mkdir(parents=True)
        compiler = '#!/bin/sh\nif [ "$1" = -dumpfullversion ]; then echo "${FAKE_GCC_VERSION:-14.2.1}"; else echo "fake compiler"; fi\n'
        for directory, names in (("gcc-bin", ("gcc", "g++")), ("amd-bin", ("amdclang", "amdclang++"))):
            for name in names:
                self.executable(self.base / directory / name, compiler)
        recorder = self.base / "record.py"
        recorder.write_text(
            'import json,os,sys\n'
            'profile=sys.argv[sys.argv.index("--profile")+1]\n'
            'with open(os.environ["TRACE"],"a") as f:\n'
            ' json.dump({"profile":profile,"argv":sys.argv[1:],"aocl":os.environ.get("AOCL_ROOT"),"rocm":os.environ.get("ROCM_PATH"),"environment":{k:os.environ.get(k) for k in ("LD_LIBRARY_PATH","LIBRARY_PATH","C_INCLUDE_PATH","CPLUS_INCLUDE_PATH")}},f);f.write("\\n")\n'
            'sys.exit(9 if os.environ.get("FAIL_PROFILE")==profile else 0)\n')
        self.executable(self.base / "base-bin/python3", '#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(recorder)) + ' "$@"\n')
        self.driver = self.base / "driver.sh"
        self.driver.write_text('''#!/bin/bash
module() {
    printf '%s\\n' "$*" >> "$MODULE_TRACE"
    case "$1" in
        purge)
            export PATH="$BASE_PATH"
            unset AOCL_ROOT ROCM_PATH LD_LIBRARY_PATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
            if [[ -n ${BASE_LD:-} ]]; then export LD_LIBRARY_PATH=$BASE_LD; fi ;;
        load)
            case "$2" in
                gcc/14.2.1)
                    export PATH="$GCC_BIN:$PATH"
                    if [[ -n $GCC_LIB_DIRS ]]; then
                        export LD_LIBRARY_PATH="$GCC_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                    fi ;;
                aocl/5.1.0/gcc14.2.1)
                    export AOCL_ROOT="$AOCL_PREFIX"
                    export LD_LIBRARY_PATH="${FAKE_AOCL_LD-$AOCL_PREFIX/lib:/usr/lib64:/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
                    export LIBRARY_PATH="$AOCL_PREFIX/lib:/usr/lib64:/usr/lib"
                    export C_INCLUDE_PATH="$AOCL_PREFIX/include" CPLUS_INCLUDE_PATH="$AOCL_PREFIX/include" ;;
                rocm/7.2.0) export PATH="$AMD_BIN:$PATH" ROCM_PATH="$ROCM_PREFIX" ;;
                *) return 8 ;;
            esac ;;
        show|-t) : ;;
        *) return 8 ;;
    esac
}
source "$SCRIPT" "$SOURCE" "$RUN_ROOT" test-build
''')

    @staticmethod
    def executable(path, contents):
        path.write_text(contents)
        path.chmod(0o700)

    def run_fake(self, **extra):
        environment = dict(os.environ)
        base_path = str(self.base / "base-bin") + ":/usr/bin:/bin"
        environment.update(GPU_SUITE_SIRIUS_BUILD_ONLY_APPROVED="yes", PATH=base_path,
            BASE_PATH=base_path, GCC_BIN=str(self.base / "gcc-bin"), AMD_BIN=str(self.base / "amd-bin"),
            AOCL_PREFIX=str(self.aocl), ROCM_PREFIX=str(self.rocm),
            BASE_LD=str(self.unrelated), GCC_LIB_DIRS=":".join(map(str, self.gcc_libraries)),
            TRACE=str(self.trace), MODULE_TRACE=str(self.modules), SCRIPT=str(SCRIPT),
            SOURCE=str(self.source), RUN_ROOT=str(self.run))
        environment.update(extra)
        return subprocess.run(["/bin/bash", str(self.driver)], env=environment,
                              capture_output=True, text=True, timeout=20)

    def records(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()] if self.trace.exists() else []

    def test_profiles_use_separate_module_environments_and_explicit_compilers(self):
        result = self.run_fake()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.records()
        self.assertEqual([r["profile"] for r in records], ["cpu", "hip", "openmp"])
        self.assertEqual(records[0]["aocl"], str(self.base / "aocl"))
        self.assertIsNone(records[0]["rocm"])
        for record in records[1:]:
            self.assertIsNone(record["aocl"])
            self.assertEqual(record["rocm"], str(self.rocm))
            self.assertIn("gfx942:sramecc+:xnack-", record["argv"])
            self.assertEqual(record["environment"]["LD_LIBRARY_PATH"], str(self.unrelated))
            for key in ("LIBRARY_PATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
                self.assertIsNone(record["environment"][key])
        for record, compiler in zip(records, ("gcc-bin/gcc", "amd-bin/amdclang", "amd-bin/amdclang")):
            argv = record["argv"]
            self.assertEqual(argv[argv.index("--c-compiler") + 1], str(self.base / compiler))
        self.assertEqual(self.modules.read_text().splitlines().count("purge"), 3)
        self.assertIn("GPU_SUITE_SIRIUS_BUILD_ONLY_COMPLETE test-build", result.stdout)
        self.assertFalse((self.run / "build-cpu").exists())

    def test_gcc_library_priority_preserves_aocl_and_other_search_paths(self):
        result = self.run_fake()
        self.assertEqual(result.returncode, 0, result.stderr)
        environment = self.records()[0]["environment"]
        gcc_paths = ":".join(map(str, self.gcc_libraries))
        aocl_paths = str(self.aocl / "lib") + ":/usr/lib64:/usr/lib"
        expected = gcc_paths + ":" + aocl_paths + ":" + str(self.unrelated)
        self.assertEqual(environment["LD_LIBRARY_PATH"], expected)
        self.assertEqual(environment["LIBRARY_PATH"], aocl_paths)
        self.assertEqual(environment["C_INCLUDE_PATH"], str(self.aocl / "include"))
        self.assertEqual(environment["CPLUS_INCLUDE_PATH"], str(self.aocl / "include"))
        self.assertIn("CPU_LD_LIBRARY_PATH_BEFORE_GCC=" + str(self.unrelated), result.stdout)
        self.assertIn("CPU_LD_LIBRARY_PATH_AFTER_GCC=" + gcc_paths + ":" + str(self.unrelated), result.stdout)
        self.assertIn("CPU_LD_LIBRARY_PATH_AFTER_AOCL=" + aocl_paths + ":" + gcc_paths, result.stdout)
        self.assertIn("CPU_GCC_PRIORITY_LIBRARY_PATH=" + gcc_paths, result.stdout)
        self.assertIn("CPU_LD_LIBRARY_PATH_EFFECTIVE=" + expected, result.stdout)

    def test_library_order_deduplicates_exact_paths_with_literal_characters(self):
        gcc_path = self.base / "toolset [literal]* space/lib"
        prefix_collision = Path(str(gcc_path) + "64")
        gcc_path.mkdir(parents=True)
        prefix_collision.mkdir()
        aocl = str(self.aocl / "lib")
        existing = str(prefix_collision) + ":" + str(self.unrelated)
        loaded = ":".join([aocl, str(gcc_path), str(prefix_collision), aocl, str(self.unrelated), str(gcc_path)])
        result = self.run_fake(BASE_LD=existing, GCC_LIB_DIRS=str(gcc_path) + ":" + str(gcc_path), FAKE_AOCL_LD=loaded)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.records()[0]["environment"]["LD_LIBRARY_PATH"],
                         ":".join([str(gcc_path), aocl, str(prefix_collision), str(self.unrelated)]))

    def test_unset_initial_library_path_is_supported(self):
        result = self.run_fake(BASE_LD="")
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = ":".join(map(str, self.gcc_libraries)) + ":" + str(self.aocl / "lib") + ":/usr/lib64:/usr/lib"
        self.assertEqual(self.records()[0]["environment"]["LD_LIBRARY_PATH"], expected)
        self.assertIsNone(self.records()[1]["environment"]["LD_LIBRARY_PATH"])

    def test_no_new_gcc_library_directory_stops_before_build_helper(self):
        result = self.run_fake(GCC_LIB_DIRS=str(self.unrelated))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GCC module introduced no usable library directories", result.stderr)
        self.assertEqual(self.records(), [])
        self.assertNotIn("load rocm/", self.modules.read_text())

    def test_missing_gcc_library_directory_stops_before_build_helper(self):
        result = self.run_fake(GCC_LIB_DIRS=str(self.base / "missing-gcc-library"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GCC module introduced a missing library directory", result.stderr)
        self.assertEqual(self.records(), [])
        self.assertNotIn("load rocm/", self.modules.read_text())

    def test_malformed_library_paths_stop_before_build_helper(self):
        for variable, stage in (("BASE_LD", "before-GCC"), ("GCC_LIB_DIRS", "after-GCC"), ("FAKE_AOCL_LD", "after-AOCL")):
            for value in ("relative/path", ":/usr/lib", "/usr/lib:", "/usr/lib::/usr/lib64"):
                with self.subTest(variable=variable, value=value):
                    result = self.run_fake(**{variable: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(stage + " contains an empty or relative LD_LIBRARY_PATH entry", result.stderr)
                    self.assertEqual(self.records(), [])
        self.assertNotIn("load rocm/", self.modules.read_text())

    def test_cpu_build_failure_stops_before_loading_gpu_modules(self):
        result = self.run_fake(FAIL_PROFILE="cpu")
        self.assertEqual(result.returncode, 9)
        self.assertEqual([r["profile"] for r in self.records()], ["cpu"])
        self.assertNotIn("load rocm/", self.modules.read_text())
        self.assertNotIn("GPU_SUITE_SIRIUS_BUILD_ONLY_COMPLETE", result.stdout)

    def test_wrong_gcc_version_stops_before_build_helper(self):
        result = self.run_fake(FAKE_GCC_VERSION="11.5.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.records(), [])
        self.assertIn("not 14.2.1", result.stderr)

    def test_missing_approval_stops_before_any_module_operation(self):
        result = self.run_fake(GPU_SUITE_SIRIUS_BUILD_ONLY_APPROVED="no")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.modules.exists())
        self.assertEqual(self.records(), [])


class RocfftComputeModuleTests(unittest.TestCase):
    """The sourced compute bootstrap uses fake modules and drivers only."""

    executable = staticmethod(RocfftModuleBuildTests.executable)

    def setUp(self):
        RocfftModuleBuildTests.setUp(self)
        for relative in ("lib", "bin", "include/hip", "include/rocfft"):
            (self.rocm / relative).mkdir(parents=True, exist_ok=True)
        for path in (self.aocl / "include/fftw3.h",
                     self.rocm / "include/hip/hip_runtime_api.h",
                     self.rocm / "include/rocfft/rocfft.h"):
            path.write_text("fake header, not compiled\n")
        self.executable(self.rocm / "bin/amdclang++",
                        '#!/bin/sh\nprintf "%s\\n" "${FAKE_AMD_VERSION:-AMD clang fake}"\n')
        self.system_libraries = [self.base / "system/lib64", self.base / "system/lib"]
        for path in self.system_libraries:
            path.mkdir(parents=True)
        # Only the two deployment constants change in this local fixture.
        # There is no production environment/CLI override for provider roots.
        self.script = self.base / "compute_modules.sh"
        canonical = (ROOT / "jobs/sirius/rocfft_compute_modules.sh").read_text()
        self.script.write_text(canonical.replace(
            "/system/apps/rhel/9.6-202602/aocl/5.1.0/gcc", str(self.aocl)).replace(
            "/opt/rocm-7.2.0", str(self.rocm)))
        self.capture = self.base / "environment.json"
        capture = self.base / "capture.py"
        capture.write_text('import json,os\nwith open(os.environ["CAPTURE"],"x") as f:\n json.dump(dict(os.environ),f)\n')
        driver = self.driver.read_text().replace(
            'unset AOCL_ROOT ROCM_PATH LD_LIBRARY_PATH',
            'unset AOCL_ROOT ROCM_PATH HIP_PATH LD_LIBRARY_PATH')
        driver = driver.replace('/usr/lib64:/usr/lib', ':'.join(map(str, self.system_libraries)))
        driver = driver.replace('export AOCL_ROOT="$AOCL_PREFIX"',
                                'export AOCL_ROOT="${FAKE_AOCL_ROOT-$AOCL_PREFIX}"')
        driver = driver.replace('export PATH="$AMD_BIN:$PATH" ROCM_PATH="$ROCM_PREFIX"',
            'export PATH="$AMD_BIN:$PATH" ROCM_PATH="${FAKE_ROCM_ROOT-$ROCM_PREFIX}" HIP_PATH="${FAKE_HIP_ROOT-$ROCM_PREFIX}"; '
            'export LD_LIBRARY_PATH="${FAKE_ROCM_LD-$ROCM_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"')
        driver = driver.replace('source "$SCRIPT" "$SOURCE" "$RUN_ROOT" test-build',
            'source "$SCRIPT" || exit $?\n' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(capture)))
        self.driver.write_text(driver)

    def run_fake(self, **extra):
        environment = dict(os.environ)
        base_path = str(self.base / "base-bin") + ":/usr/bin:/bin"
        environment.update(GPU_SUITE_SIRIUS_COMPUTE_APPROVED="yes", PBS_JOBID="123.test",
            STAGE31_SOURCE_ROOT=str(self.source), STAGE31_RUN_ROOT=str(self.run), STAGE31_RUN_ID="test-build",
            PATH=base_path, BASE_PATH=base_path, GCC_BIN=str(self.base / "gcc-bin"),
            AMD_BIN=str(self.rocm / "bin"), AOCL_PREFIX=str(self.aocl), ROCM_PREFIX=str(self.rocm),
            BASE_LD=str(self.unrelated), GCC_LIB_DIRS=":".join(map(str, self.gcc_libraries)),
            TRACE=str(self.trace), MODULE_TRACE=str(self.modules), SCRIPT=str(self.script),
            CAPTURE=str(self.capture), HOME=str(self.base / "historical-home"))
        environment.update(extra)
        return subprocess.run(["/bin/bash", str(self.driver)], env=environment,
                              capture_output=True, text=True, timeout=20)

    def test_compute_modules_export_selected_drivers_and_preserve_all_libraries(self):
        result = self.run_fake()
        self.assertEqual(result.returncode, 0, result.stderr)
        environment = json.loads(self.capture.read_text())
        expected = self.gcc_libraries + [self.rocm / "lib", self.aocl / "lib"] + self.system_libraries + [self.unrelated]
        self.assertEqual(environment["LD_LIBRARY_PATH"], ":".join(map(str, expected)))
        self.assertEqual(environment["CPU_CC"], str(self.base / "gcc-bin/gcc"))
        self.assertEqual(environment["CPU_CXX"], str(self.base / "gcc-bin/g++"))
        self.assertEqual(environment["HIP_CLANG"], str(self.rocm / "bin/amdclang++"))
        self.assertEqual(environment["OPENMP_CXX"], environment["HIP_CLANG"])
        self.assertEqual(environment["FFTW3f_ROOT"], str(self.aocl))
        self.assertEqual(environment["FFTW3f_THREADING"], "OPENMP")
        self.assertEqual(environment["HOME"], str(self.base / "historical-home"))
        self.assertEqual([line for line in self.modules.read_text().splitlines() if line.startswith("load ")],
                         ["load gcc/14.2.1", "load aocl/5.1.0/gcc14.2.1", "load rocm/7.2.0"])
        self.assertIn("COMPUTE_FINAL_ENVIRONMENT HOME=", result.stdout)
        self.assertIn("COMPUTE_MODULE_ENVIRONMENT USER=", result.stdout)
        self.assertIn("COMPUTE_FINAL_MODULES", result.stdout)
        self.assertFalse(self.trace.exists(), "must not invoke the build recorder")
        self.assertFalse((self.run / "compute").exists(), "must not invoke the job body")

    def test_compute_rocm_replacement_retains_gcc_aocl_and_initial_paths(self):
        result = self.run_fake(FAKE_ROCM_LD=str(self.rocm / "lib"))
        self.assertEqual(result.returncode, 0, result.stderr)
        environment = json.loads(self.capture.read_text())
        expected = self.gcc_libraries + [self.rocm / "lib", self.aocl / "lib"] + self.system_libraries + [self.unrelated]
        self.assertEqual(environment["LD_LIBRARY_PATH"], ":".join(map(str, expected)))

    def test_compute_requires_approval_allocation_and_run_gates_before_modules(self):
        for values in ({"GPU_SUITE_SIRIUS_COMPUTE_APPROVED": "no"}, {"PBS_JOBID": ""},
                       {"STAGE31_RUN_ID": ""}, {"STAGE31_SOURCE_ROOT": ""},
                       {"STAGE31_RUN_ROOT": str(self.base)}, {"STAGE31_RUN_ID": "other"}):
            with self.subTest(values=values):
                result = self.run_fake(**values)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.modules.exists())
                self.assertFalse(self.capture.exists())

    def test_compute_rejects_unidentifiable_gcc_library_contribution(self):
        result = self.run_fake(GCC_LIB_DIRS=str(self.unrelated))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("introduced no usable library directories", result.stderr)
        self.assertNotIn("load aocl/", self.modules.read_text())
        self.assertFalse(self.capture.exists())

    def test_compute_rejects_malformed_or_missing_library_paths_without_fallback(self):
        for variable in ("BASE_LD", "GCC_LIB_DIRS", "FAKE_AOCL_LD", "FAKE_ROCM_LD"):
            for value in ("relative", str(self.unrelated) + ":", ":" + str(self.unrelated),
                          str(self.base / "missing-library")):
                with self.subTest(variable=variable, value=value):
                    result = self.run_fake(**{variable: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("library directory", result.stderr)
                    self.assertFalse(self.capture.exists())

    def test_compute_rejects_wrong_compiler_or_provider_without_fallback(self):
        for values in ({"FAKE_GCC_VERSION": "11.5.0"}, {"FAKE_AOCL_ROOT": str(self.base)},
                       {"FAKE_ROCM_ROOT": str(self.base)}, {"FAKE_HIP_ROOT": str(self.base)},
                       {"FAKE_AMD_VERSION": "ordinary clang"}):
            with self.subTest(values=values):
                result = self.run_fake(**values)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.trace.exists())

    def test_compute_environment_changes_are_limited_to_child_shell(self):
        before = dict(os.environ)
        result = self.run_fake()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(dict(os.environ), before)

    def test_compute_helper_must_be_sourced(self):
        result = subprocess.run(["/bin/bash", str(self.script)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("source this helper", result.stderr)
        self.assertFalse(self.modules.exists())


if __name__ == "__main__":
    unittest.main()
