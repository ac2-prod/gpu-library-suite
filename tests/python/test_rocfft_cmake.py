"""Host-only CMake checks; synthetic FFTW never demonstrates ROCm support."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc") and shutil.which("ar"),
                     "CMake, C compiler and archiver are required")
class RocfftCmakeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rocfft-cmake-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = os.environ.copy()
        for key in ("ROCM_PATH", "HIP_PATH", "FFTW_ROOT", "FFTW3_ROOT", "FFTW3F_ROOT",
                    "FFTW3f_ROOT", "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH",
                    "CMAKE_LIBRARY_PATH", "PKG_CONFIG_PATH", "PKG_CONFIG_LIBDIR"):
            self.environment.pop(key, None)

    def command(self, args, success=True):
        result = subprocess.run([str(item) for item in args], capture_output=True,
                                text=True, env=self.environment, check=False)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def configure(self, *options, success=True, name="build"):
        build = self.root / name
        args = ["cmake", "-S", ROOT, "-B", build,
                "-DGPU_SUITE_VENDOR=amd", "-DGPU_SUITE_AMD_LIBRARIES=rocfft", "-DGPU_SUITE_BUILD_TESTS=OFF",
                "-DGPU_SUITE_BUILD_CUDA=OFF", "-DGPU_SUITE_BUILD_OPENACC=OFF",
                "-DGPU_SUITE_BUILD_HIP=OFF", "-DGPU_SUITE_BUILD_OPENMP=OFF",
                "-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=ON"]
        result = self.command(args + list(options), success=success)
        return build, result

    def test_cpu_only_without_optional_dependencies_builds_common(self):
        build, result = self.configure()
        self.assertIn("rocFFT CPU targets", result.stdout)
        self.assertIn("FFTW3f discovery explicitly disabled", result.stdout)
        self.command(["cmake", "--build", build, "--target", "gpu_suite_common"])
        metadata = json.loads((build / "build-metadata.json").read_text())
        self.assertEqual(metadata["vendor"], "amd")
        self.assertEqual(metadata["build_profile"], "cpu")
        self.assertEqual(metadata["dependencies"], {})
        self.assertEqual(metadata["hip"]["compiler"], "")
        self.assertNotIn("cuda", metadata)

    def fixture(self, backends=("pthread",), name="synthetic-fftw", incomplete=False,
                base_requires_threads=False):
        prefix = self.root / name
        (prefix / "include").mkdir(parents=True)
        (prefix / "lib").mkdir()
        shutil.copyfile(ROOT / "tests/fixtures/fftw/include/fftw3.h", prefix / "include/fftw3.h")
        base = prefix / "fftw3f.o"
        base_source = ROOT / "tests/fixtures/fftw/fake_fftw3f.c"
        if base_requires_threads:
            text = base_source.read_text().replace('#include "fftw3.h"',
                '#include "fftw3.h"\nint synthetic_pthread_runtime(void);')
            text = text.replace('  struct gpu_suite_fake_fftw_plan *plan;',
                '  struct gpu_suite_fake_fftw_plan *plan;\n'
                '  if (synthetic_pthread_runtime() == 0) return NULL;')
            base_source = prefix / "combined-base.c"
            base_source.write_text(text)
            runtime = prefix / "pthread-runtime.c"
            runtime.write_text("int synthetic_pthread_runtime(void) { return 1; }\n")
            self.command(["cc", "-c", runtime, "-o", prefix / "pthread-runtime.o"])
            self.command(["ar", "rcs", prefix / "lib/libsynthetic_pthread.a", prefix / "pthread-runtime.o"])
            modules = prefix / "cmake"
            modules.mkdir()
            (modules / "FindThreads.cmake").write_text(
                'set(Threads_FOUND TRUE)\n'
                'set(CMAKE_THREAD_LIBS_INIT "' + str(prefix / "lib/libsynthetic_pthread.a") + '")\n'
                'if(NOT TARGET Threads::Threads)\n'
                '  add_library(Threads::Threads INTERFACE IMPORTED)\n'
                '  set_target_properties(Threads::Threads PROPERTIES\n'
                '    INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")\n'
                'endif()\n')
        self.command(["cc", "-c", base_source, "-I", prefix / "include", "-o", base])
        self.command(["ar", "rcs", prefix / "lib/libfftw3f.a", base])
        for backend in backends:
            source = (ROOT / "tests/fixtures/fftw/fake_fftw3f_threads.c").read_text()
            if incomplete is True or incomplete == backend:
                source = source.replace("fftwf_cleanup_threads", "missing_cleanup_threads")
            if backend == "openmp":
                source = "int synthetic_openmp_runtime(void);\n" + source
                source = source.replace("return 1;", "return synthetic_openmp_runtime();")
                runtime = prefix / "runtime.c"
                runtime.write_text("int synthetic_openmp_runtime(void) { return 1; }\n")
                self.command(["cc", "-c", runtime, "-o", prefix / "runtime.o"])
                self.command(["ar", "rcs", prefix / "lib/libruntime.a", prefix / "runtime.o"])
                modules = prefix / "cmake"
                modules.mkdir(exist_ok=True)
                (modules / "FindOpenMP.cmake").write_text(
                    'set(OpenMP_C_FOUND TRUE)\nset(OpenMP_FOUND TRUE)\n'
                    'set(OpenMP_C_FLAGS "-DSYNTHETIC_OPENMP_RUNTIME=1")\n'
                    'set(OpenMP_C_LIBRARIES "' + str(prefix / "lib/libruntime.a") + '")\n'
                    'add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)\n'
                    'set_target_properties(OpenMP::OpenMP_C PROPERTIES\n'
                    '  INTERFACE_COMPILE_OPTIONS "${OpenMP_C_FLAGS}"\n'
                    '  INTERFACE_LINK_LIBRARIES "${OpenMP_C_LIBRARIES}")\n')
            source_file = prefix / (backend + ".c")
            source_file.write_text(source)
            obj = prefix / (backend + ".o")
            self.command(["cc", "-c", source_file, "-I", prefix / "include", "-o", obj])
            library = {"pthread": "fftw3f_threads", "openmp": "fftw3f_omp",
                       "combined": "fftw3f"}[backend]
            self.command(["ar", "rcs", prefix / ("lib/lib" + library + ".a"), obj])
        return prefix

    def find_fftw(self, name, *options):
        source = self.root / (name + "-source")
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(fftw_discovery_regression LANGUAGES C)\n'
            'list(APPEND CMAKE_MODULE_PATH "' + str(ROOT / "cmake") + '")\n'
            'set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)\n'
            'set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH FALSE)\n'
            'set(CMAKE_FIND_USE_INSTALL_PREFIX FALSE)\n'
            'set(CMAKE_FIND_USE_PACKAGE_ROOT_PATH FALSE)\n'
            'find_package(FFTW3f QUIET)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/discovery.txt" '
            '"${FFTW3f_FOUND}\\n${FFTW3f_THREADS_FOUND}\\n${FFTW3f_INCLUDE_DIR}\\n'
            '${FFTW3f_LIBRARY}\\n${FFTW3f_THREADS_LIBRARY}\\n${FFTW3f_DISABLE_REASON}\\n'
            '${FFTW3f_THREAD_BACKEND}\\n${FFTW3f_THREADS_DISABLE_REASON}\\n")\n')
        build = self.root / name
        self.command(["cmake", "-S", source, "-B", build] + list(options))
        return (build / "discovery.txt").read_text().splitlines()

    def test_fftw_package_roots_and_environment_hints_resolve_real_libraries(self):
        prefix = self.fixture()
        for number, root_name in enumerate(("FFTW3f_ROOT", "FFTW3F_ROOT", "FFTW3_ROOT", "FFTW_ROOT")):
            for source in ("cmake", "environment"):
                with self.subTest(root=root_name, source=source):
                    options = ["-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON"]
                    if source == "cmake":
                        options.append("-D" + root_name + "=" + str(prefix))
                    else:
                        self.environment[root_name] = str(prefix)
                    try:
                        found = self.find_fftw("root-" + str(number) + "-" + source, *options)
                    finally:
                        self.environment.pop(root_name, None)
                    self.assertEqual(found[:2], ["TRUE", "TRUE"])
                    self.assertEqual(found[2], str(prefix / "include"))
                    self.assertEqual(found[3], str(prefix / "lib/libfftw3f.a"))
                    self.assertEqual(found[4], str(prefix / "lib/libfftw3f_threads.a"))
                    self.assertEqual(found[5], "")

    @unittest.skipUnless(shutil.which("pkg-config"), "pkg-config is required")
    def test_fftw_pkg_config_hints_resolve_nonstandard_include_and_library_dirs(self):
        prefix = self.fixture()
        include = self.root / "package-headers"
        libraries = self.root / "package-libraries"
        include.mkdir()
        libraries.mkdir()
        shutil.copyfile(prefix / "include/fftw3.h", include / "fftw3.h")
        for name in ("fftw3f", "fftw3f_threads"):
            shutil.copyfile(prefix / ("lib/lib" + name + ".a"),
                            libraries / ("lib" + name + ".a"))
        metadata = self.root / "pkgconfig"
        metadata.mkdir()
        (metadata / "fftw3f.pc").write_text(
            "Name: synthetic single precision FFTW\nDescription: discovery fixture only\n"
            "Version: 3.3.0\nLibs: -L" + str(libraries) + " -lfftw3f\n"
            "Cflags: -I" + str(include) + "\n")
        self.environment["PKG_CONFIG_LIBDIR"] = str(metadata)
        found = self.find_fftw("pkg-config", "-DPKG_CONFIG_EXECUTABLE=" + shutil.which("pkg-config"))
        self.assertEqual(found[:2], ["TRUE", "TRUE"])
        self.assertEqual(found[2], str(include))
        self.assertEqual(found[3], str(libraries / "libfftw3f.a"))
        self.assertEqual(found[4], str(libraries / "libfftw3f_threads.a"))

    def test_missing_fftw_reports_discovery_failure_without_claiming_link_failure(self):
        found = self.find_fftw("missing-fftw", "-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=ON")
        self.assertEqual(found[:2], ["FALSE", "FALSE"])
        self.assertIn("fftw3.h and fftw3f library not found", found[5])
        self.assertIn("FFTW3f_ROOT", found[5])
        self.assertNotIn("compile-and-link probe failed", found[5])

    def test_synthetic_fftw_resolves_cpu_targets_and_dependency_hashes(self):
        prefix = self.fixture()
        build, _ = self.configure("-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=OFF",
                                  "-DFFTW_ROOT=" + str(prefix))
        self.command(["cmake", "--build", build, "--target", "fft_cpu", "fft_cpu_bench"])
        metadata = json.loads((build / "build-metadata.json").read_text())
        self.assertEqual(set(metadata["dependencies"]), {"fftw3f", "fftw3f_threads"})
        for name in ("fftw3f", "fftw3f_threads"):
            path = prefix / ("lib/lib" + name + ".a")
            self.assertEqual(metadata["dependencies"][name]["path"], str(path))
            self.assertEqual(metadata["dependencies"][name]["sha256"],
                             hashlib.sha256(path.read_bytes()).hexdigest())
        descriptor = json.loads((build / "partial-manifest-input.json").read_text())
        self.assertEqual({entry["target_name"] for entry in descriptor["entries"]},
                         {"fft_cpu", "fft_cpu_bench"})
        for entry in descriptor["entries"]:
            self.assertEqual(entry["library"], "rocfft")
            self.assertEqual(entry["implementation"], "cpu")
            self.assertEqual(entry["compiler_language"], "c")
            self.assertEqual(entry["build_profile"], "cpu")

    def test_thread_backends_link_complete_api_and_record_selected_provenance(self):
        for backend in ("pthread", "openmp", "combined"):
            with self.subTest(backend=backend):
                prefix = self.fixture((backend,), name=backend)
                options = ["-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=OFF",
                           "-DFFTW3f_ROOT=" + str(prefix),
                           "-DFFTW3f_THREADING=" + backend.upper()]
                if backend == "openmp":
                    options.append("-DCMAKE_MODULE_PATH=" + str(prefix / "cmake"))
                build, _ = self.configure(*options, name="build-" + backend)
                self.command(["cmake", "--build", build, "--target", "fft_cpu_bench"])
                metadata = json.loads((build / "build-metadata.json").read_text())
                threading = metadata["fftw_threading"]
                self.assertEqual(threading["requested"], backend.upper())
                self.assertEqual(threading["backend"], backend)
                expected = {"fftw3f"}
                if backend == "pthread":
                    expected.add("fftw3f_threads")
                if backend == "openmp":
                    expected.update(("fftw3f_omp", "fftw_openmp_runtime_0"))
                    self.assertEqual(threading["compile_options"], ["-DSYNTHETIC_OPENMP_RUNTIME=1"])
                    self.assertEqual(metadata["dependencies"]["fftw_openmp_runtime_0"]["path"],
                                     str(prefix / "lib/libruntime.a"))
                    self.assertIn("-DSYNTHETIC_OPENMP_RUNTIME=1",
                                  (build / "amd/c-cpp/rocfft/CMakeFiles/fft_cpu_bench.dir/flags.make").read_text())
                self.assertEqual(set(threading["dependency_names"]), expected)
                self.assertEqual(set(metadata["dependencies"]), expected)
                for dependency in metadata["dependencies"].values():
                    self.assertEqual(dependency["sha256"],
                                     hashlib.sha256(Path(dependency["path"]).read_bytes()).hexdigest())
                descriptor = json.loads((build / "partial-manifest-input.json").read_text())
                benchmark = next(item for item in descriptor["entries"] if item["target_name"] == "fft_cpu_bench")
                self.assertIn("cpu-fftw-threaded", benchmark["cpu_backends"])

    def test_static_combined_base_runtime_enables_serial_and_threaded_targets(self):
        prefix = self.fixture(("combined",), base_requires_threads=True)
        for mode in ("AUTO", "COMBINED"):
            with self.subTest(mode=mode):
                build, _ = self.configure("-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=OFF",
                    "-DFFTW3f_ROOT=" + str(prefix), "-DFFTW3f_THREADING=" + mode,
                    "-DCMAKE_MODULE_PATH=" + str(prefix / "cmake"), name="static-" + mode)
                self.command(["cmake", "--build", build, "--target", "fft_cpu", "fft_cpu_bench"])
                metadata = json.loads((build / "build-metadata.json").read_text())
                threading = metadata["fftw_threading"]
                self.assertEqual(threading["backend"], "combined")
                self.assertEqual(set(threading["dependency_names"]), {"fftw3f", "fftw_pthread_runtime_0"})
                runtime = prefix / "lib/libsynthetic_pthread.a"
                self.assertEqual(threading["link_options"], [str(runtime)])
                self.assertEqual(metadata["dependencies"]["fftw_pthread_runtime_0"],
                    {"path": str(runtime), "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest()})

    def test_static_combined_incomplete_api_and_other_explicit_modes_disable_base(self):
        prefix = self.fixture(("combined",), name="complete", base_requires_threads=True)
        incomplete = self.fixture(("combined",), name="incomplete", incomplete=True,
                                  base_requires_threads=True)
        for mode, provider in (("PTHREAD", prefix), ("OPENMP", prefix), ("NONE", prefix),
                               ("AUTO", incomplete), ("COMBINED", incomplete)):
            with self.subTest(mode=mode):
                found = self.find_fftw("static-rejected-" + mode,
                    "-DFFTW3f_ROOT=" + str(provider), "-DFFTW3f_THREADING=" + mode,
                    "-DCMAKE_MODULE_PATH=" + str(provider / "cmake"),
                    "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY")
                self.assertEqual(found[:2], ["FALSE", "FALSE"])
                self.assertEqual(found[6], "none")
                self.assertIn("compile-and-link probe failed", found[5])

    def test_auto_threading_prefers_pthread_then_openmp_then_combined(self):
        for index, (backends, expected) in enumerate(((('pthread', 'openmp', 'combined'), 'pthread'),
                                                     (('openmp', 'combined'), 'openmp'),
                                                     (('combined',), 'combined'))):
            with self.subTest(expected=expected):
                prefix = self.fixture(backends, name="auto-" + str(index))
                options = ["-DFFTW3f_ROOT=" + str(prefix), "-DFFTW3f_THREADING=AUTO"]
                if "openmp" in backends:
                    options.append("-DCMAKE_MODULE_PATH=" + str(prefix / "cmake"))
                found = self.find_fftw("auto-build-" + str(index), *options)
                self.assertEqual(found[:2], ["TRUE", "TRUE"])
                self.assertEqual(found[6], expected)

    def test_auto_threading_skips_a_library_with_incomplete_api(self):
        prefix = self.fixture(("pthread", "openmp"), incomplete="pthread")
        found = self.find_fftw("incomplete-auto", "-DFFTW3f_ROOT=" + str(prefix),
            "-DCMAKE_MODULE_PATH=" + str(prefix / "cmake"))
        self.assertEqual(found[:2], ["TRUE", "TRUE"])
        self.assertEqual(found[6], "openmp")

    def test_invalid_threading_selection_is_rejected_even_without_cpu_discovery(self):
        for enabled in ("ON", "OFF"):
            with self.subTest(discovery_disabled=enabled):
                _, result = self.configure("-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=" + enabled,
                    "-DFFTW3f_THREADING=invalid", name="invalid-" + enabled, success=False)
                self.assertIn("FFTW3f_THREADING must be", result.stderr)

    def test_explicit_missing_backend_and_none_never_fall_back(self):
        prefix = self.fixture(("combined",))
        for mode in ("PTHREAD", "OPENMP", "NONE"):
            with self.subTest(mode=mode):
                found = self.find_fftw("no-fallback-" + mode,
                    "-DFFTW3f_ROOT=" + str(prefix), "-DFFTW3f_THREADING=" + mode)
                self.assertEqual(found[:2], ["TRUE", "FALSE"])
                self.assertEqual(found[6], "none")
                self.assertTrue(found[7])

    def test_incomplete_thread_api_is_rejected_for_every_backend(self):
        for backend in ("pthread", "openmp", "combined"):
            with self.subTest(backend=backend):
                prefix = self.fixture((backend,), name="incomplete-" + backend, incomplete=True)
                options = ["-DFFTW3f_ROOT=" + str(prefix),
                           "-DFFTW3f_THREADING=" + backend.upper(),
                           "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"]
                if backend == "openmp":
                    options.append("-DCMAKE_MODULE_PATH=" + str(prefix / "cmake"))
                found = self.find_fftw("incomplete-build-" + backend, *options)
                self.assertEqual(found[:2], ["TRUE", "FALSE"])
                self.assertIn("all three thread APIs did not compile and link", found[7])

    def test_thread_library_from_another_provider_is_rejected(self):
        prefix = self.fixture((), name="base-provider")
        other = self.fixture(("pthread", "openmp"), name="other-provider")
        for mode, variable, library in (("PTHREAD", "THREADS", "fftw3f_threads"),
                                         ("OPENMP", "OPENMP", "fftw3f_omp")):
            with self.subTest(mode=mode):
                found = self.find_fftw("mixed-" + mode, "-DFFTW3f_ROOT=" + str(prefix),
                    "-DFFTW3f_THREADING=" + mode,
                    "-DFFTW3f_" + variable + "_LIBRARY=" + str(other / ("lib/lib" + library + ".a")))
                self.assertEqual(found[:2], ["TRUE", "FALSE"])
                self.assertIn("outside the selected base provider directories", found[7])

    def test_openmp_candidate_requires_linkable_runtime_and_records_no_unselected_library(self):
        prefix = self.fixture(("pthread", "openmp"))
        module = prefix / "cmake/FindOpenMP.cmake"
        module.write_text(module.read_text().replace('"${OpenMP_C_LIBRARIES}"', '""'))
        found = self.find_fftw("broken-runtime", "-DFFTW3f_ROOT=" + str(prefix),
            "-DFFTW3f_THREADING=OPENMP", "-DCMAKE_MODULE_PATH=" + str(prefix / "cmake"))
        self.assertEqual(found[:2], ["TRUE", "FALSE"])
        self.assertIn("did not compile and link", found[7])
        build, _ = self.configure("-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=OFF",
            "-DFFTW3f_ROOT=" + str(prefix), "-DFFTW3f_THREADING=NONE",
            "-DFFTW3f_THREADS_LIBRARY=" + str(prefix / "lib/libfftw3f_threads.a"))
        metadata = json.loads((build / "build-metadata.json").read_text())
        self.assertEqual(set(metadata["dependencies"]), {"fftw3f"})
        self.assertEqual(metadata["fftw_threading"]["dependency_names"], ["fftw3f"])
        self.assertEqual(metadata["fftw_threading"]["backend"], "none")

    def test_headers_without_fftw_symbols_do_not_enable_cpu_target(self):
        prefix = self.fixture()
        empty = self.root / "unrelated.c"
        empty.write_text("void unrelated_symbol(void) {}\n")
        self.command(["cc", "-c", empty, "-o", self.root / "unrelated.o"])
        self.command(["ar", "rcs", self.root / "unrelated.a", self.root / "unrelated.o"])
        build, result = self.configure("-DCMAKE_DISABLE_FIND_PACKAGE_FFTW3f=OFF",
            "-DFFTW3f_INCLUDE_DIR=" + str(prefix / "include"),
            "-DFFTW3f_LIBRARY=" + str(self.root / "unrelated.a"),
            "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY")
        self.assertIn("FFTW3f plan/execute compile-and-link probe failed", result.stdout)
        self.assertFalse((build / "partial-manifest-input.json").exists())

    def test_amd_profiles_cannot_be_combined(self):
        for options in (("-DGPU_SUITE_BUILD_HIP=ON",),
                        ("-DGPU_SUITE_BUILD_OPENMP=ON",),
                        ("-DGPU_SUITE_BUILD_CPU=OFF", "-DGPU_SUITE_BUILD_HIP=ON",
                         "-DGPU_SUITE_BUILD_OPENMP=ON")):
            with self.subTest(options=options):
                _, result = self.configure(*options, success=False)
                self.assertIn("separate CPU, HIP and OpenMP build trees", result.stderr)

    def test_vendor_and_profile_changes_require_new_build_trees(self):
        build, _ = self.configure()
        result = self.command(["cmake", "-S", ROOT, "-B", build,
                               "-DGPU_SUITE_VENDOR=nvidia"], success=False)
        self.assertIn("new build tree when changing vendor", result.stderr)
        _, result = self.configure("-DGPU_SUITE_BUILD_CPU=OFF",
            "-DGPU_SUITE_BUILD_HIP=ON", success=False)
        self.assertIn("new build tree when changing AMD profile", result.stderr)

    def test_hip_without_architecture_disables_without_compiler_detection(self):
        for architecture in ("", "native", "gfx900;native"):
            with self.subTest(architecture=architecture):
                build, result = self.configure("-DGPU_SUITE_BUILD_CPU=OFF",
                    "-DGPU_SUITE_BUILD_HIP=ON", "-DCMAKE_HIP_ARCHITECTURES=" + architecture,
                    name="hip-" + str(len(architecture)))
                self.assertIn("supply explicit CMAKE_HIP_ARCHITECTURES", result.stdout)
                self.assertNotIn("Looking for a HIP compiler", result.stdout)
                self.assertFalse((build / "partial-manifest-input.json").exists())
                metadata = json.loads((build / "build-metadata.json").read_text())
                self.assertEqual(metadata["hip"]["architectures"], architecture)

    def test_hipcc_wrapper_is_rejected(self):
        _, result = self.configure("-DGPU_SUITE_BUILD_CPU=OFF", "-DGPU_SUITE_BUILD_HIP=ON",
            "-DCMAKE_HIP_COMPILER=/unavailable/hipcc", success=False)
        self.assertIn("Clang, not the hipcc wrapper", result.stderr)

    def test_openmp_without_architecture_disables_only_requested_targets(self):
        build, result = self.configure("-DGPU_SUITE_BUILD_CPU=OFF",
            "-DGPU_SUITE_BUILD_OPENMP=ON")
        self.assertIn("explicit GPU_SUITE_AMD_GPU_TARGET", result.stdout)
        self.command(["cmake", "--build", build, "--target", "gpu_suite_common"])

    def test_api_link_probes_and_cpp_hip_language_are_explicit(self):
        module = (ROOT / "cmake/GpuSuiteAmd.cmake").read_text()
        probe = (ROOT / "cmake/FindRocFFT.cmake").read_text()
        targets = (ROOT / "amd/c-cpp/rocfft/CMakeLists.txt").read_text()
        self.assertIn("gpu_suite_check_hip_link", module)
        self.assertIn("gpu_suite_check_cxx_link", module)
        self.assertNotIn("try_run", module)
        for symbol in ("rocfft_plan_create", "rocfft_execute", "rocfft_plan_get_work_buffer_size",
                       "rocfft_execution_info_set_work_buffer", "hipDeviceSynchronize"):
            self.assertIn(symbol, probe)
        self.assertIn("PROPERTIES LANGUAGE HIP", targets)
        self.assertIn("map(to:input[0:8]) map(from:output[0:8])", module)
        self.assertIn("use_device_ptr(input, output)", module)

    def test_default_nvidia_metadata_format_is_unchanged(self):
        build = self.root / "nvidia"
        self.command(["cmake", "-S", ROOT, "-B", build, "-DGPU_SUITE_BUILD_TESTS=OFF",
                      "-DGPU_SUITE_BUILD_CPU=OFF"])
        metadata = json.loads((build / "build-metadata.json").read_text())
        self.assertEqual(metadata["build_profile"], "cpu-cuda")
        self.assertNotIn("vendor", metadata)
        self.assertNotIn("hip", metadata)
        self.assertEqual(set(metadata), {"build_profile", "build_type", "c", "cxx", "cuda",
                                         "git_commit", "git_dirty", "git_metadata_available",
                                         "nvhpc", "openacc"})


if __name__ == "__main__":
    unittest.main()
