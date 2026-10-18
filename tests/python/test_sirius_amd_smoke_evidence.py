"""Local file/trace fixtures only: no real loader, agent, SSH, or GPU calls."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "jobs/sirius"))
SPEC = importlib.util.spec_from_file_location(
    "sirius_amd_smoke_evidence", ROOT / "jobs/sirius/amd_smoke_evidence.py")
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


class SiriusAmdSmokeEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="amd-smoke-evidence-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.program = self.file("program", b"\x7fELFfake benchmark")
        self.loader = self.file("ld-linux.so", b"\x7fELFfake loader")
        self.depfile = self.base / "link.d"

    def file(self, name, content=None):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if content is not None else b"\x7fELF" + name.encode())
        return path

    @staticmethod
    def identity(path):
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def entry(self, family="rocsparse", profile="cpu", role="benchmark",
              variant="aocl-sparse", backends=None):
        return {"library": family, "implementation": profile, "build_profile": profile,
                "executable_role": role, "backend_variant": variant,
                "supported_cpu_backends": ["cpu-aocl-sparse"] if backends is None else backends,
                "executable_path": str(self.program),
                "binary_sha256": self.identity(self.program)["sha256"]}

    def cpu_metadata(self):
        # Keys, ordering, backend variants and support tokens mirror the saved
        # Sirius CPU build metadata. Bytes/paths are exclusively local fakes.
        files = {name: self.file(name + ".so") for name in (
            "fftw3f", "fftw3f_omp", "fftw_openmp_runtime_0", "cblas", "lapacke", "aocl_sparse")}
        files["fftw_openmp_runtime_1"] = self.file("libpthread.a", b"!<arch>\n")
        files["lapacke_link_1"] = files["cblas"]
        files["aocl_sparse_header"] = self.file("include_LP64/aoclsparse.h", b"typedef int aoclsparse_int;\n")
        dependencies = {name: self.identity(path) for name, path in files.items()}
        return {"vendor": "amd", "build_profile": "cpu", "dependencies": dependencies,
                "fftw_threading": {"requested": "OPENMP", "backend": "openmp",
                    "compile_options": ["-fopenmp"], "link_options": [],
                    "dependency_names": ["fftw3f", "fftw_openmp_runtime_0", "fftw_openmp_runtime_1", "fftw3f_omp"]},
                "amd_build_inputs": {
                    "GPU_SUITE_CPU_BLAS_BACKEND": "GENERIC_CBLAS",
                    "GPU_SUITE_CPU_LAPACK_BACKEND": "GENERIC_LAPACKE",
                    "GPU_SUITE_CPU_SPARSE_BACKEND": "AOCLSPARSE",
                    "GPU_SUITE_CPU_THRUST_BACKEND": "STL",
                    "CBLAS_INTERFACE_LINK_LIBRARIES": str(files["cblas"]) + ";Threads::Threads;dl;m",
                    "LAPACKE_INTERFACE_LINK_LIBRARIES": str(files["lapacke"]) + ";" + str(files["cblas"]) + ";Threads::Threads;dl;m",
                    "AOCLSparse_INTERFACE_LINK_LIBRARIES": str(files["aocl_sparse"]) + ";Threads::Threads;dl;m",
                    "AOCLSparse_INTERFACE_INCLUDE_DIRECTORIES": str(files["aocl_sparse_header"].parent)}}

    def gpu_metadata(self, profile="hip"):
        files = {name: self.file(name + ".so") for name in (
            "hip", "rocfft", "rocblas", "rocsparse", "rocsolver", "rocrand")}
        files["rocthrust"] = self.file("thrust/version.h", b"#define THRUST_VERSION 200000\n")
        files["rocprim"] = self.file("rocprim/rocprim.hpp", b"#pragma once\n")
        return {"vendor": "amd", "build_profile": profile,
                "dependencies": {name: self.identity(path) for name, path in files.items()},
                "amd_build_inputs": {}}

    def links(self, paths, target=None):
        self.depfile.write_text((target or self.program.name) + ": \\\n  "
                                + " \\\n  ".join(str(path) for path in paths) + "\n")
        return self.depfile

    def ldd(self, paths):
        return "linux-vdso.so.1 (0x0000ffff)\n" + "".join(
            "lib{0}.so => {1} (0x123456)\n".format(index, path)
            for index, path in enumerate(paths)) + str(self.loader) + " (0x4321)\n"

    def trace(self, paths, pid=431, late=(), main_path="", without_fini=()):
        prefix = "     " + str(pid) + ":\t"
        records = ["find library=libfake.so [0]; searching",
                   " search path=/fake/glibc-hwcaps/x86-64-v4:/fake (RUNPATH from file " + str(self.program) + ")",
                   "  trying file=/fake/libfake.so", " search cache=/etc/ld.so.cache", ""]
        records += ["calling init: " + str(path) for path in paths]
        records += ["initialize program: " + str(self.program), "",
                    "transferring control: " + str(self.program), ""]
        records += ["calling init: " + str(path) for path in late]
        records += ["calling fini: " + main_path + " [0]"]
        records += ["calling fini: " + str(path) + " [0]" for path in reversed(list(paths) + list(late))
                    if path not in without_fini]
        return "\n".join(prefix + record for record in records) + "\nbenchmark warning (non-loader)\n"

    def sparse_inputs(self):
        entry, metadata = self.entry(), self.cpu_metadata()
        primary = Path(metadata["dependencies"]["aocl_sparse"]["path"])
        self.links([primary])
        return entry, metadata, primary

    def observation_checks(self, entry, metadata, loaded):
        return (
            ("resolution", lambda: evidence.validate_artifact_resolution(
                entry, metadata, self.ldd(loaded), self.depfile)),
            ("trace", lambda: evidence.validate_loaded_trace(
                entry, metadata, self.trace([self.loader] + loaded),
                str(self.program), 431, self.depfile)),
        )

    def test_saved_cpu_metadata_selects_each_family_and_both_roles(self):
        metadata = self.cpu_metadata()
        rows = [
            ("rocblas", "generic_cblas", ["cpu-generic-cblas"], {"cblas"}),
            ("rocsolver", "generic_lapacke", ["cpu-generic-lapacke"], {"lapacke", "lapacke_link_1"}),
            ("rocsparse", "aocl-sparse", ["cpu-aocl-sparse"], {"aocl_sparse", "aocl_sparse_header"}),
            ("rocrand", "std-random-serial", ["cpu-std-random-serial"], set()),
            ("rocthrust", "stl-serial", ["cpu-stl-serial"], set()),
        ]
        for role in ("example", "benchmark"):
            for family, variant, backends, expected in rows:
                with self.subTest(role=role, family=family):
                    entry = self.entry(family=family, role=role, variant=variant, backends=backends)
                    self.assertEqual(set(evidence.select_artifact_dependencies(entry, metadata)), expected)
        example = self.entry("rocfft", role="example", variant="fftw3f-serial", backends=["cpu-fftw-serial"])
        benchmark = self.entry("rocfft", variant="fftw3f-serial+threaded", backends=["cpu-fftw-serial", "cpu-fftw-threaded"])
        self.assertEqual(set(evidence.select_artifact_dependencies(example, metadata)), {"fftw3f"})
        self.assertEqual(set(evidence.select_artifact_dependencies(benchmark, metadata)),
                         set(metadata["fftw_threading"]["dependency_names"]))

    def test_gpu_families_profiles_and_header_only_inputs(self):
        for profile in ("hip", "openmp"):
            metadata = self.gpu_metadata(profile)
            for role in ("example", "benchmark"):
                for family in sorted(evidence.FAMILIES):
                    with self.subTest(profile=profile, role=role, family=family):
                        entry = self.entry(family, profile, role, profile + "-" + family, [])
                        selected = evidence.select_artifact_dependencies(entry, metadata)
                        names = {"hip", family}
                        if family == "rocsolver": names.add("rocblas")
                        if family == "rocthrust": names.add("rocprim")
                        self.assertEqual(set(selected), names)
                        for name in names:
                            self.assertEqual(selected[name]["kind"],
                                             "header" if name in {"rocthrust", "rocprim"} else "elf")

    def test_fftw_pthread_combined_serial_and_legacy_selection(self):
        entry = self.entry("rocfft", variant="fftw3f-serial+threaded", backends=["cpu-fftw-serial", "cpu-fftw-threaded"])
        for backend in ("pthread", "combined", "legacy"):
            base = self.file("libfftw.so")
            dependencies = {"fftw3f": self.identity(base)}
            if backend != "combined": dependencies["fftw3f_threads"] = self.identity(self.file("libfftw_threads.so"))
            metadata = {"vendor": "amd", "build_profile": "cpu", "dependencies": dependencies}
            if backend != "legacy":
                metadata["fftw_threading"] = {"requested": backend.upper(), "backend": backend,
                    "compile_options": [], "link_options": [], "dependency_names": list(dependencies)}
            with self.subTest(backend=backend):
                self.assertEqual(set(evidence.select_artifact_dependencies(entry, metadata)), set(dependencies))
        entry["executable_role"] = "example"
        with self.assertRaises(ValueError): evidence.select_artifact_dependencies(entry, metadata)

    def test_explicit_onemkl_openblas_reference_and_openmp_cpu_selection(self):
        metadata = self.cpu_metadata()
        for family, option, base in (("rocblas", "GPU_SUITE_CPU_BLAS_BACKEND", "cblas"),
                                     ("rocsolver", "GPU_SUITE_CPU_LAPACK_BACKEND", "lapacke")):
            for provider in ("OPENBLAS", "ONEMKL"):
                metadata["amd_build_inputs"][option] = provider
                entry = self.entry(family, variant=provider.lower(), backends=["cpu-" + provider.lower()])
                self.assertIn(base, evidence.select_artifact_dependencies(entry, metadata))
        metadata["amd_build_inputs"]["GPU_SUITE_CPU_SPARSE_BACKEND"] = "REFERENCE"
        reference = self.entry(variant="reference-csr", backends=["cpu-reference-csr"])
        self.assertEqual(evidence.select_artifact_dependencies(reference, metadata), {})
        metadata["dependencies"]["mkl_sparse"] = self.identity(self.file("mkl.so"))
        metadata["amd_build_inputs"].update(GPU_SUITE_CPU_SPARSE_BACKEND="ONEMKL",
                                           MKLSparse_INTERFACE_LINK_LIBRARIES=metadata["dependencies"]["mkl_sparse"]["path"] + ";Threads::Threads;dl;m")
        self.assertEqual(set(evidence.select_artifact_dependencies(
            self.entry(variant="onemkl-sparse", backends=["cpu-onemkl"]), metadata)), {"mkl_sparse"})
        metadata["amd_build_inputs"]["GPU_SUITE_CPU_THRUST_BACKEND"] = "OPENMP"
        metadata["dependencies"]["cpu_openmp_runtime_0"] = self.identity(self.file("omp.so"))
        omp = self.entry("rocthrust", variant="openmp", backends=["cpu-openmp"])
        self.assertEqual(set(evidence.select_artifact_dependencies(omp, metadata)), {"cpu_openmp_runtime_0"})

    def test_unrelated_metadata_does_not_become_required_runtime(self):
        entry, metadata, primary = self.sparse_inputs()
        metadata["dependencies"]["rocblas"] = {"path": "/unrelated/nonexistent", "sha256": "0" * 64}
        selected = evidence.validate_artifact_resolution(entry, metadata, self.ldd([primary]), self.depfile)
        self.assertEqual(set(selected["dependencies"]), {"aocl_sparse", "aocl_sparse_header"})
        self.assertNotIn("rocblas", selected["resolved_objects"])

    def test_missing_changed_alias_and_header_misclassification_rejected(self):
        entry, metadata, primary = self.sparse_inputs()
        for mutation in ("missing", "hash", "alias", "header-elf", "bad-profile", "bad-role", "bad-backend"):
            altered, changed = copy.deepcopy(metadata), dict(entry)
            if mutation == "missing": altered["dependencies"].pop("aocl_sparse")
            elif mutation == "hash": altered["dependencies"]["aocl_sparse"]["sha256"] = "0" * 64
            elif mutation == "alias": altered["dependencies"]["aocl_sparse"]["path"] = str(self.base) + "/./" + primary.name
            elif mutation == "header-elf": altered["dependencies"]["aocl_sparse_header"] = self.identity(primary)
            elif mutation == "bad-profile": changed["build_profile"] = "hip"
            elif mutation == "bad-role": changed["executable_role"] = "future"
            else: altered["amd_build_inputs"]["GPU_SUITE_CPU_SPARSE_BACKEND"] = "ONEMKL"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evidence.select_artifact_dependencies(changed, altered)
        header = Path(metadata["dependencies"]["aocl_sparse_header"]["path"])
        header.write_bytes(b"\x7fELFnot a header")
        metadata["dependencies"]["aocl_sparse_header"] = self.identity(header)
        with self.assertRaises(ValueError): evidence.select_artifact_dependencies(entry, metadata)

    def test_ordered_extras_require_complete_index_and_interface_binding(self):
        entry, metadata, _ = self.sparse_inputs()
        extra = self.file("extra.a", b"!<arch>\nfake")
        metadata["dependencies"]["aocl_sparse_link_2"] = self.identity(extra)
        inputs = metadata["amd_build_inputs"]
        inputs["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = metadata["dependencies"]["aocl_sparse"]["path"] + ";Threads::Threads;" + str(extra) + ";dl;m"
        self.assertEqual(list(evidence.select_artifact_dependencies(entry, metadata)),
                         ["aocl_sparse", "aocl_sparse_link_2", "aocl_sparse_header"])
        for mutation in ("missing", "wrong-index", "reordered", "extra-unresolved"):
            altered = copy.deepcopy(metadata)
            if mutation == "missing": altered["dependencies"].pop("aocl_sparse_link_2")
            elif mutation == "wrong-index": altered["dependencies"]["aocl_sparse_link_1"] = altered["dependencies"].pop("aocl_sparse_link_2")
            elif mutation == "reordered": altered["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = str(extra) + ";" + metadata["dependencies"]["aocl_sparse"]["path"]
            else: altered["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] += ";guessed-provider"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evidence.select_artifact_dependencies(entry, altered)

    def test_ldd_resolution_requires_elf_paths_and_exact_selected_identity(self):
        entry, metadata, primary = self.sparse_inputs()
        alias = str(primary.parent) + "/./" + primary.name
        result = evidence.validate_artifact_resolution(entry, metadata, self.ldd([alias]), self.depfile)
        self.assertEqual(result["kind"], "ldd-resolution")
        self.assertEqual(result["resolved_objects"][str(primary)], self.identity(primary)["sha256"])
        self.assertNotIn("initialized_objects", result)
        for text in ("", "libfake.so => not found\n", "garbage\n", self.ldd([self.program]),
                     self.ldd([Path(metadata["dependencies"]["aocl_sparse_header"]["path"])])):
            with self.subTest(text=text), self.assertRaises(ValueError):
                evidence.validate_artifact_resolution(entry, metadata, text, self.depfile)

    def test_actual_link_depfile_is_required_and_bound_to_artifact_inputs(self):
        entry, metadata, primary = self.sparse_inputs()
        with self.assertRaises(ValueError): evidence.validate_artifact_resolution(entry, metadata, self.ldd([primary]))
        for paths, target in (([self.loader], None), ([primary], "another_binary")):
            self.links(paths, target)
            with self.assertRaises(ValueError):
                evidence.validate_artifact_resolution(entry, metadata, self.ldd([primary]), self.depfile)
        self.links([primary], str(self.program))
        result = evidence.validate_artifact_resolution(entry, metadata, self.ldd([primary]), self.depfile)
        self.assertEqual(result["link_dependency_file"]["sha256"], self.identity(self.depfile)["sha256"])

    def test_openmp_depfile_historical_objects_for_six_families_and_roles(self):
        metadata = self.gpu_metadata("openmp")
        archive = self.file("build/common/c-cpp/libgpu_suite_common.a", b"!<arch>\nfake common")
        relative_archive = "../../../common/c-cpp/libgpu_suite_common.a"
        for family in sorted(evidence.FAMILIES):
            for role in ("example", "benchmark"):
                self.program = self.file("build/amd/c-cpp/" + family + "/openmp_" + family + "_" + role)
                entry = self.entry(family, "openmp", role, "openmp-" + family, [])
                selected = evidence.select_artifact_dependencies(entry, metadata)
                libraries = [Path(item["path"]) for item in selected.values() if item["kind"] == "elf"]
                objects = [str(self.base / "tmp" / (self.program.name + ".openmp.image.wrapper-5a1183.o")),
                           str(self.base / "toolchain/crtbegin.o"),
                           "CMakeFiles/" + self.program.name + ".dir/benchmarks/input.cpp.o"]
                if family == "rocthrust":
                    objects.append("CMakeFiles/wrapper.dir/examples/rocthrust_wrapper.hip.o")
                tokens = objects + [str(path) for path in libraries] + [relative_archive]
                self.links(tokens)
                self.depfile.write_text(self.depfile.read_text() + "\n" +
                                        "\n\n".join(token + ":" for token in tokens) + "\n")
                self.assertFalse((self.base / "tmp").exists())
                self.assertFalse((self.program.parent / "CMakeFiles").exists())
                for route, validate in self.observation_checks(entry, metadata, libraries):
                    with self.subTest(family=family, role=role, route=route), mock.patch.object(
                            evidence, "_file_identity", wraps=evidence._file_identity) as read:
                        result = validate()
                        link = result["link_dependency_file"]
                        self.assertEqual(link["raw_prerequisites"], tokens)
                        self.assertEqual(link["build_directory"], str(self.program.parent))
                        self.assertEqual(link["prerequisites"][:len(objects)], [
                            {"token": token, "kind": "historical-object", "current_file_verified": False}
                            for token in objects])
                        verified = [item for item in link["prerequisites"] if item["current_file_verified"]]
                        self.assertEqual({item["path"] for item in verified},
                                         {str(path) for path in libraries} | {str(archive)})
                        self.assertEqual(verified[-1]["sha256"], self.identity(archive)["sha256"])
                        self.assertFalse(any(str(call.args[0]).endswith(".o") for call in read.call_args_list))
                        self.assertEqual(link["sha256"], self.identity(self.depfile)["sha256"])

    def test_depfile_relative_archive_uses_artifact_directory_and_keeps_alias_tokens(self):
        self.program = self.file("build/amd/c-cpp/rocsparse/program")
        entry, metadata, _ = self.sparse_inputs()
        archive = self.file("build/common/c-cpp/libprovider.a", b"!<arch>\nprovider")
        metadata["dependencies"]["aocl_sparse"] = self.identity(archive)
        metadata["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = str(archive)
        relative = "../../../common/c-cpp/libprovider.a"
        alias = str(archive.parent) + "/../c-cpp/" + archive.name
        self.links([relative, alias, relative])
        self.depfile.write_text(self.depfile.read_text() + "\n" + relative + ":\n" + relative + ":\n")
        for route, validate in self.observation_checks(entry, metadata, []):
            with self.subTest(route=route):
                result = validate()
                link = result["link_dependency_file"]
                self.assertEqual(link["raw_prerequisites"], [relative, alias, relative])
                self.assertEqual([item["path"] for item in link["prerequisites"]], [str(archive)] * 3)
                self.assertEqual([item["sha256"] for item in link["prerequisites"]],
                                 [self.identity(archive)["sha256"]] * 3)
                self.assertEqual(result["dependencies"]["aocl_sparse"]["kind"], "static-archive")

    def test_depfile_missing_persistent_inputs_are_rejected_by_both_observation_routes(self):
        entry, metadata, primary = self.sparse_inputs()
        for missing in (str(self.base / "missing.so"), str(self.base / "missing.so.1"),
                        str(self.base / "missing.a"), str(self.base / "missing.ld"),
                        "../never-provided/library.a"):
            self.links([self.base / "missing-wrapper.o", primary, missing])
            for route, validate in self.observation_checks(entry, metadata, [primary]):
                with self.subTest(missing=missing, route=route), self.assertRaises((ValueError, OSError)):
                    validate()
        self.links([primary, self.base])
        for route, validate in self.observation_checks(entry, metadata, [primary]):
            with self.subTest(nonregular=True, route=route), self.assertRaises(ValueError):
                validate()

    def test_depfile_malformed_make_and_target_mismatch_are_rejected(self):
        entry, metadata, primary = self.sparse_inputs()
        first = self.program.name + ": " + str(primary)
        malformed = ["", "\n" + first, "other: " + str(primary), self.program.name + ":",
                     first + " $(UNREVIEWED)", first + " escaped\\ name.o", first + " #comment",
                     first + "\x00", first + "\r\n", first + "\nother:\n",
                     first + "\n" + str(primary) + ": extra.o\n", first + "\n\techo bad\n",
                     first + "\n" + self.program.name + ": " + str(primary) + "\n"]
        for text in malformed:
            self.depfile.write_text(text)
            for route, validate in self.observation_checks(entry, metadata, [primary]):
                with self.subTest(text=text, route=route), self.assertRaises(ValueError):
                    validate()

    def test_historical_object_tokens_cannot_satisfy_configured_or_script_inputs(self):
        entry, metadata, _ = self.sparse_inputs()
        # Even an existing ELF-looking .o cannot masquerade as a current library.
        obj = self.file("object.o")
        metadata["dependencies"]["aocl_sparse"] = self.identity(obj)
        metadata["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = str(obj)
        self.links([obj])
        for route, validate in self.observation_checks(entry, metadata, [obj]):
            with self.subTest(kind="configured", route=route), self.assertRaises(ValueError):
                validate()
        script = self.file("libscript.so", ("INPUT ( " + str(obj) + " )\n").encode())
        metadata["dependencies"]["aocl_sparse"] = self.identity(script)
        metadata["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = str(script)
        self.links([script, obj])
        for route, validate in self.observation_checks(entry, metadata, [obj]):
            with self.subTest(kind="linker-script", route=route), self.assertRaises(ValueError):
                validate()

    def test_depfile_objects_do_not_hide_archive_ambiguity_or_missing_runtime(self):
        entry, metadata, _ = self.sparse_inputs()
        dynamic = self.file("libprovider.so.1")
        first = self.file("one/libsupport.a", b"!<arch>\none")
        second = self.file("two/libsupport.a", b"!<arch>\ntwo")
        script = self.file("libprovider.so", ("INPUT ( " + str(dynamic) + " -lsupport )\n").encode())
        metadata["dependencies"]["aocl_sparse"] = self.identity(script)
        metadata["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = str(script)
        for libraries, loaded, reason in (([script, dynamic, first, second], [dynamic], "ambiguous"),
                                          ([script, dynamic, first], [], "unobserved-runtime"),
                                          ([script, dynamic], [dynamic], "missing-archive")):
            self.links([self.base / "tmp/image.wrapper.o"] + libraries)
            for route, validate in self.observation_checks(entry, metadata, loaded):
                with self.subTest(reason=reason, route=route), self.assertRaises(ValueError):
                    validate()

    def test_dependency_free_cpu_still_records_supplied_depfile(self):
        metadata = self.cpu_metadata()
        for family, variant, backends in (("rocrand", "std-random-serial", ["cpu-std-random-serial"]),
                                          ("rocthrust", "stl-serial", ["cpu-stl-serial"])):
            for role in ("example", "benchmark"):
                entry = self.entry(family, role=role, variant=variant, backends=backends)
                token = "CMakeFiles/program.dir/input.cpp.o"
                self.links([token])
                for route, validate in self.observation_checks(entry, metadata, []):
                    with self.subTest(family=family, role=role, route=route):
                        result = validate()
                        self.assertEqual(result["dependencies"], {})
                        self.assertEqual(result["link_dependency_file"]["raw_prerequisites"], [token])
                        self.assertFalse(result["link_dependency_file"]["prerequisites"][0]["current_file_verified"])

    def native_hip_inputs(self, family="rocblas", role="benchmark"):
        """Real Sirius native-HIP link pattern, using only small local files."""
        metadata = self.gpu_metadata()
        compiler = str(self.base / "amdclang++")
        metadata["hip"] = {"compiler": "Clang", "compiler_path": compiler}
        self.program = self.file("native/" + family + "/program")
        directory = self.program.parent
        entry = self.entry(family, "hip", role, "hip-" + family, [])
        entry.update(target_name=self.program.name, compiler_language="hip", compiler="Clang")
        selected = evidence.select_artifact_dependencies(entry, metadata)
        obj = self.file("native/" + family + "/CMakeFiles/program.dir/benchmarks/input.cpp.o")
        common = self.file("native/common/libgpu_suite_common.a", b"!<arch>\nlocal archive")
        hip = metadata["dependencies"]["hip"]["path"]
        libraries = [item["path"] for name, item in selected.items() if name != "hip" and item["kind"] != "header"]
        argv = [compiler, "-O3", "-DNDEBUG", "--offload-arch=gfx942:sramecc+:xnack-", "--hip-link",
                "--rtlib=compiler-rt", "-unwindlib=libgcc", str(obj.relative_to(directory)),
                "-o", self.program.name, "-L" + str(self.base / "toolset/lib64")] + libraries + [hip,
                "../common/" + common.name, "-lm", hip, "-lgcc", "-lgcc"]
        link = directory / "CMakeFiles/program.dir/link.txt"
        link.write_text(shlex.join(argv) + "\n")
        stdout = self.file("native/" + family + "/build.stdout", (
            "[100%] Linking HIP executable program\ncd " + str(directory)
            + " && /fixture/cmake -E cmake_link_script CMakeFiles/program.dir/link.txt --verbose=1\n"
            + shlex.join(argv) + "\n[100%] Built target program\n").encode())
        descriptor = {"mode": "native-hip-link-command", "link_command": self.identity(link),
                      "build_stdout": self.identity(stdout), "build_directory": str(directory)}
        return entry, metadata, descriptor, argv

    def test_native_hip_recorded_link_without_depfile_for_six_families_and_roles(self):
        for family in sorted(evidence.FAMILIES):
            for role in ("example", "benchmark"):
                entry, metadata, descriptor, argv = self.native_hip_inputs(family, role)
                with self.subTest(family=family, role=role):
                    self.assertFalse((Path(descriptor["link_command"]["path"]).parent / "link.d").exists())
                    proof = evidence.validate_link_evidence(entry, metadata, descriptor)
                    link = proof["link_dependency_file"]
                    self.assertEqual(link["mode"], "native-hip-link-command")
                    self.assertEqual(link["argv"], argv)
                    hip = metadata["dependencies"]["hip"]["path"]
                    self.assertEqual(sum(item["path"] == hip for item in link["ordered_inputs"]), 2)
                    self.assertEqual([item["argv_index"] for item in link["ordered_inputs"]],
                                     sorted(item["argv_index"] for item in link["ordered_inputs"]))
                    loaded = [Path(item["path"]) for item in proof["dependencies"].values() if item["kind"] == "elf"]
                    resolution = evidence.validate_artifact_resolution(entry, metadata, self.ldd(loaded), descriptor)
                    trace = evidence.validate_loaded_trace(entry, metadata, self.trace([self.loader] + loaded),
                                                          str(self.program), 431, descriptor)
                    self.assertEqual(resolution["link_dependency_file"], link)
                    self.assertEqual(trace["link_dependency_file"], link)
                    with self.assertRaises(ValueError):
                        evidence.validate_loaded_trace(entry, metadata, self.trace([self.loader]), str(self.program), 431, descriptor)

    def test_native_hip_explicit_mode_does_not_fallback_from_missing_depfile(self):
        entry, metadata, descriptor, _ = self.native_hip_inputs()
        with self.assertRaises(ValueError): evidence.validate_link_evidence(entry, metadata, self.depfile)
        with self.assertRaises(ValueError): evidence.validate_link_evidence(entry, metadata, None)
        for profile in ("cpu", "openmp"):
            altered = dict(entry, implementation=profile, build_profile=profile)
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                evidence.validate_link_evidence(altered, metadata, descriptor)
        altered = dict(descriptor, mode="auto")
        with self.assertRaises(ValueError): evidence.validate_link_evidence(entry, metadata, altered)

    def test_native_hip_hash_missing_symlink_and_target_identity_rejections(self):
        entry, metadata, descriptor, _ = self.native_hip_inputs()
        for name in ("link_command", "build_stdout"):
            altered = copy.deepcopy(descriptor); altered[name]["sha256"] = "0" * 64
            with self.subTest(name=name), self.assertRaises(ValueError):
                evidence.validate_link_evidence(entry, metadata, altered)
            altered[name]["path"] = str(self.base / "missing")
            with self.assertRaises((ValueError, OSError)):
                evidence.validate_link_evidence(entry, metadata, altered)
        with mock.patch.object(Path, "is_symlink", return_value=True), self.assertRaises(ValueError):
            evidence.validate_link_evidence(entry, metadata, descriptor)
        for changes in ({"target_name": "other"}, {"compiler_language": "cxx"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evidence.validate_link_evidence(dict(entry, **changes), metadata, descriptor)
        for changes in ({"build_directory": str(self.base)}, {"link_command": None}, {"extra": "unapproved"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evidence.validate_link_evidence(entry, metadata, dict(descriptor, **changes))

    def test_native_hip_link_stdout_mismatch_duplicate_and_success_marker_rejected(self):
        entry, metadata, descriptor, argv = self.native_hip_inputs()
        stdout = Path(descriptor["build_stdout"]["path"])
        original = stdout.read_text()
        variants = [original.replace("-O3", "-O2"), original + shlex.join(argv) + "\n",
                    original.replace("[100%] Built target program\n", ""),
                    "[100%] Built target program\n" + original.replace("[100%] Built target program\n", ""),
                    original + "[100%] Built target program\n"]
        for changed in variants:
            stdout.write_text(changed)
            descriptor["build_stdout"] = self.identity(stdout)
            with self.subTest(text=changed), self.assertRaises(ValueError):
                evidence.validate_link_evidence(entry, metadata, descriptor)

    def test_native_hip_optional_dependency_output_preserves_both_raw_commands(self):
        entry, metadata, descriptor, argv = self.native_hip_inputs()
        pair = ["-Xlinker", "--dependency-file=CMakeFiles/program.dir/link.d"]
        extended = argv[:7] + pair + argv[7:]
        for recorded, built in ((argv, extended), (extended, argv), (extended, extended)):
            for key, command in (("link_command", recorded), ("build_stdout", built)):
                path = Path(descriptor[key]["path"])
                path.write_text(shlex.join(command) + "\n" +
                                ("[100%] Built target program\n" if key == "build_stdout" else ""))
                descriptor[key] = self.identity(path)
            with self.subTest(recorded=recorded, built=built):
                link = evidence.validate_link_evidence(entry, metadata, descriptor)["link_dependency_file"]
                self.assertEqual(link["argv"], recorded)
                self.assertEqual(link["build_argv"], built)
                self.assertEqual(link["comparison_argv"], argv)
                for item in link["ordered_inputs"]:
                    self.assertEqual(recorded[item["argv_index"]], item["token"])

    def test_native_hip_dependency_output_does_not_hide_other_changes(self):
        entry, metadata, descriptor, argv = self.native_hip_inputs()
        pair = ["-Xlinker", "--dependency-file=CMakeFiles/program.dir/link.d"]
        variants = [pair + pair, ["-Xlinker"], ["-Xlinker", "--as-needed"],
                    ["-Xlinker", "--dependency-file=CMakeFiles/other.dir/link.d"],
                    ["-Xlinker", "--dependency-file=../link.d"],
                    ["-Wl,--dependency-file=CMakeFiles/program.dir/link.d"]]
        commands = [argv + extra for extra in variants]
        commands += [["-O2" if arg == "-O3" else arg for arg in argv] + pair,
                     argv[:-1] + pair, argv + pair + ["-lchanged"]]
        output_index = argv.index("-o") + 1
        commands.append(argv[:output_index] + pair + argv[output_index:])
        stdout = Path(descriptor["build_stdout"]["path"])
        for built in commands:
            stdout.write_text(shlex.join(built) + "\n[100%] Built target program\n")
            descriptor["build_stdout"] = self.identity(stdout)
            with self.subTest(built=built), self.assertRaises(ValueError):
                evidence.validate_link_evidence(entry, metadata, descriptor)

    def test_native_hip_response_shell_compiler_output_and_input_guesses_rejected(self):
        entry, metadata, descriptor, argv = self.native_hip_inputs()
        link, stdout = (Path(descriptor[key]["path"]) for key in ("link_command", "build_stdout"))
        changes = [argv + ["@response.rsp"], argv + ["-Wl,@response.rsp"], argv + ["$(touch/never)"],
                   argv + [";", "echo"], ["/other/clang++"] + argv[1:],
                   ["other" if value == self.program.name else value for value in argv],
                   [value for value in argv if value != metadata["dependencies"]["rocblas"]["path"]],
                   argv + ["--unknown-input=guess"], argv + ["ambiguous input.o"]]
        for changed in changes:
            command = shlex.join(changed)
            link.write_text(command + "\n"); stdout.write_text(command + "\n[100%] Built target program\n")
            descriptor.update(link_command=self.identity(link), build_stdout=self.identity(stdout))
            with self.subTest(argv=changed), self.assertRaises(ValueError):
                evidence.validate_link_evidence(entry, metadata, descriptor)

    def test_native_hip_alias_library_inputs_resolve_and_preserve_logged_tokens(self):
        entry, metadata, descriptor, argv = self.native_hip_inputs()
        library = metadata["dependencies"]["rocblas"]["path"]
        alias = str(Path(library).parent) + "/./" + Path(library).name
        argv = [alias if item == library else item for item in argv]
        for key in ("link_command", "build_stdout"):
            path = Path(descriptor[key]["path"])
            path.write_text(shlex.join(argv) + ("\n[100%] Built target program\n" if key == "build_stdout" else "\n"))
            descriptor[key] = self.identity(path)
        link = evidence.validate_link_evidence(entry, metadata, descriptor)["link_dependency_file"]
        self.assertEqual(next(item for item in link["ordered_inputs"] if item["token"] == alias)["path"], library)

    def test_static_archive_binds_link_input_without_fake_dso_requirement(self):
        entry, metadata, _ = self.sparse_inputs()
        archive = self.file("libaocl.a", b"!<arch>\nfake archive")
        metadata["dependencies"]["aocl_sparse"] = self.identity(archive)
        metadata["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = str(archive) + ";Threads::Threads;dl;m"
        self.links([archive])
        result = evidence.validate_loaded_trace(entry, metadata, self.trace([self.loader]), str(self.program), 431, self.depfile)
        self.assertEqual(result["dependencies"]["aocl_sparse"]["kind"], "static-archive")
        self.assertEqual(result["initialized_objects"], [self.identity(self.loader)])

    def test_linker_script_requires_recorded_dynamic_and_static_inputs(self):
        entry, metadata, _ = self.sparse_inputs()
        dynamic, archive = self.file("libprovider.so.1"), self.file("libsupport.a", b"!<arch>\n")
        script = self.file("libprovider.so", ("/* GNU ld */\nINPUT ( " + str(dynamic) + " AS_NEEDED ( -lsupport ) )\n").encode())
        metadata["dependencies"]["aocl_sparse"] = self.identity(script)
        metadata["amd_build_inputs"]["AOCLSparse_INTERFACE_LINK_LIBRARIES"] = str(script)
        self.links([script, dynamic, archive])
        result = evidence.validate_artifact_resolution(entry, metadata, self.ldd([dynamic]), self.depfile)
        self.assertEqual([v["kind"] for v in result["dependencies"]["aocl_sparse"]["inputs"]], ["elf", "static-archive"])
        self.links([script, dynamic])
        with self.assertRaises(ValueError): evidence.validate_artifact_resolution(entry, metadata, self.ldd([dynamic]), self.depfile)
        duplicate = self.file("another/libsupport.a", b"!<arch>\n")
        self.links([script, dynamic, archive, duplicate])
        with self.assertRaises(ValueError): evidence.validate_artifact_resolution(entry, metadata, self.ldd([dynamic]), self.depfile)
        self.links([script, dynamic, archive])
        with self.assertRaises(ValueError): evidence.validate_artifact_resolution(entry, metadata, self.ldd([]), self.depfile)

    def test_unknown_or_nested_linker_script_content_rejected(self):
        entry, metadata, primary = self.sparse_inputs()
        for content in (b"not an ELF library", b"SEARCH_DIR(/guessed) INPUT(-lfake)", b"INPUT(../../escape)"):
            primary.write_bytes(content)
            metadata["dependencies"]["aocl_sparse"] = self.identity(primary)
            with self.subTest(content=content), self.assertRaises(ValueError):
                evidence.select_artifact_dependencies(entry, metadata)

    def test_trace_lifecycle_late_loads_and_raw_stderr_are_separate(self):
        plugin = self.file("libomptarget.rtl.amdgpu.so")
        trace = self.trace([self.loader], late=[plugin])
        result = evidence.parse_ld_debug_trace(trace, str(self.program), 431)
        self.assertEqual(result["initialized_paths"], sorted([str(self.loader), str(plugin)]))
        self.assertEqual(result["application_stderr"], "benchmark warning (non-loader)")
        self.assertEqual(result["trace_sha256"], hashlib.sha256(trace.encode()).hexdigest())
        self.assertNotIn("/fake/libfake.so", result["initialized_paths"])
        self.assertEqual(result["pid"], 431)

    def test_trace_sirius_shape_records_five_initializations_and_three_object_finalizations(self):
        # Sanitized ordering of the saved Sirius CPU BLAS trace: loader/libc
        # initialization, three other DSOs, named main fini, only three DSO fini.
        metadata = self.cpu_metadata()
        entry = self.entry("rocblas", role="example", variant="generic_cblas", backends=["cpu-generic-cblas"])
        primary = Path(metadata["dependencies"]["cblas"]["path"])
        libc, pthread, libm = (self.file(name) for name in ("libc.so.6", "libpthread.so.0", "libm.so.6"))
        initialized = [self.loader, libc, pthread, libm, primary]
        trace = self.trace(initialized, main_path=str(self.program), without_fini=[self.loader, libc])
        self.links([primary])
        result = evidence.validate_loaded_trace(entry, metadata, trace, str(self.program), 431, self.depfile)
        parsed = result["process"]
        self.assertEqual(parsed["initialized_paths"], sorted(str(path) for path in initialized))
        self.assertEqual(parsed["observed_finalization_paths"], sorted(map(str, [pthread, libm, primary])))
        self.assertEqual(parsed["initializations_without_observed_finalization"], sorted(map(str, [self.loader, libc])))
        self.assertTrue(parsed["main_finalization_observed"])
        self.assertEqual(sum(event["event"] == "calling init" for event in parsed["events"]), 5)
        self.assertEqual(sum(event["event"] == "calling fini" for event in parsed["events"]), 4)
        self.assertEqual(result["initialized_objects"], sorted(map(self.identity, initialized), key=lambda item: item["path"]))
        self.assertIn("process exit is checked separately", parsed["scope"])
        self.assertIn("not a failure or proof of teardown", parsed["scope"])
        self.assertNotIn("exit", parsed)

    def test_trace_unmatched_initialization_does_not_depend_on_library_name(self):
        entry, metadata, primary = self.sparse_inputs()
        neutral = self.file("arbitrary-provider-without-fini.so")
        trace = self.trace([self.loader, primary, neutral], without_fini=[self.loader, primary, neutral])
        result = evidence.validate_loaded_trace(entry, metadata, trace, str(self.program), 431, self.depfile)
        self.assertEqual(result["process"]["observed_finalization_paths"], [])
        self.assertEqual(result["process"]["initializations_without_observed_finalization"],
                         sorted(map(str, [self.loader, primary, neutral])))
        self.assertEqual(result["dependencies"]["aocl_sparse"]["sha256"], self.identity(primary)["sha256"])

    def test_trace_missing_required_initialization_is_not_inferred_from_search_or_fini(self):
        entry, metadata, primary = self.sparse_inputs()
        trace = self.trace([self.loader], without_fini=[self.loader])
        trace = "431: trying file=" + str(primary) + "\n" + trace
        parsed = evidence.parse_ld_debug_trace(trace, str(self.program), 431)
        self.assertNotIn(str(primary), parsed["initialized_paths"])
        with self.assertRaisesRegex(ValueError, "required ELF path/hash not observed"):
            evidence.validate_loaded_trace(entry, metadata, trace, str(self.program), 431, self.depfile)
        with self.assertRaisesRegex(ValueError, "finalization has no initialization"):
            evidence.parse_ld_debug_trace(trace + "431: calling fini: " + str(primary) + " [0]\n",
                                          str(self.program), 431)

    def test_trace_partial_finalization_still_rejects_duplicate_and_out_of_order_events(self):
        primary = self.file("provider.so")
        trace = self.trace([self.loader, primary], without_fini=[self.loader])
        variants = [trace + "431: calling fini: " + str(primary) + " [0]\n",
                    trace + "431: calling init: " + str(self.loader) + "\n",
                    trace + "431: calling fini:  [0]\n",
                    "431: calling fini:  [0]\n" + trace,
                    trace.replace("431:\tinitialize program: " + str(self.program) + "\n", ""),
                    trace.replace("431:\ttransferring control: " + str(self.program) + "\n", ""),
                    trace.replace("431:\tcalling fini:  [0]\n", ""),
                    trace + "999: calling init: /other/provider.so\n"]
        for altered in variants:
            with self.subTest(trace=altered), self.assertRaises(ValueError):
                evidence.parse_ld_debug_trace(altered, str(self.program), 431)

    def test_trace_destructor_loading_and_reload_preserve_observation_only_accounting(self):
        plugin = self.file("plugin-without-finalizer.so")
        trace = self.trace([self.loader], without_fini=[self.loader])
        first_instance = "431: calling init: {0}\n431: calling fini: {0} [0]\n".format(plugin)
        # A second instance may initialize during main's destructors without a
        # fini log event. Observed fini for the first instance does not close it.
        trace = first_instance + trace + "431: calling init: " + str(plugin) + "\n"
        result = evidence.parse_ld_debug_trace(trace, str(self.program), 431)
        self.assertEqual(result["observed_finalization_paths"], [str(plugin)])
        self.assertEqual(result["initializations_without_observed_finalization"], sorted(map(str, [self.loader, plugin])))
        self.assertEqual(sum(event["event"] == "calling init" and event["path"] == str(plugin)
                             for event in result["events"]), 2)
        self.assertTrue(result["main_finalization_observed"])

    def test_trace_reloads_and_constructor_or_destructor_loading_are_supported(self):
        path = self.file("plugin.so")
        trace = self.trace([self.loader])
        constructor = "431: calling init: {0}\n431: calling fini: {0} [0]\n".format(path)
        trace = constructor + trace
        # A destructor may initialize and release a plugin after main fini.
        trace = trace.replace("431:\tcalling fini:  [0]\n", "431:\tcalling fini:  [0]\n" + constructor)
        result = evidence.parse_ld_debug_trace(trace, str(self.program), 431)
        self.assertIn(str(path), result["initialized_paths"])
        full_main = self.trace([self.loader], main_path=str(self.program))
        self.assertEqual(evidence.parse_ld_debug_trace(full_main, str(self.program))["pid"], 431)

    def test_trace_missing_duplicate_wrong_pid_program_and_unknown_records_rejected(self):
        trace = self.trace([self.loader])
        mutations = ["", trace.replace("initialize program", "unknown initialization"),
                     trace.replace("431:", "432:", 1),
                     trace.replace("transferring control: " + str(self.program), "transferring control: /other/program"),
                     trace.replace("431:\tcalling fini:  [0]\n", ""),
                     trace + "431: unrecognized trace syntax\n",
                     trace + "432: calling init: /other/library.so\n",
                     trace + "calling init: /missing-pid.so\n",
                     trace + "431: transferring control: " + str(self.program) + "\n",
                     trace.replace("431:\tcalling init: " + str(self.loader), "431:\tcalling init: relative.so")]
        for altered in mutations:
            with self.subTest(trace=altered), self.assertRaises(ValueError):
                evidence.parse_ld_debug_trace(altered, str(self.program), 431)

    def test_trace_binding_rejects_binary_or_provider_change_and_wrong_library(self):
        entry, metadata, primary = self.sparse_inputs()
        trace = self.trace([self.loader, primary])
        result = evidence.validate_loaded_trace(entry, metadata, trace, str(self.program), 431, self.depfile)
        self.assertEqual(result["initialized_objects"], sorted([self.identity(self.loader), self.identity(primary)], key=lambda x: x["path"]))
        self.assertIn("not symbol ownership or worker count", result["scope"])
        bad = dict(entry, binary_sha256="0" * 64)
        with self.assertRaises(ValueError): evidence.validate_loaded_trace(bad, metadata, trace, str(self.program), 431, self.depfile)
        with self.assertRaises(ValueError): evidence.validate_loaded_trace(entry, metadata, self.trace([self.loader]), str(self.program), 431, self.depfile)
        with self.assertRaises(ValueError): evidence.validate_loaded_trace(entry, metadata, trace, "/other/program", 431, self.depfile)
        primary.write_bytes(b"\x7fELFchanged after build")
        with self.assertRaises(ValueError): evidence.validate_loaded_trace(entry, metadata, trace, str(self.program), 431, self.depfile)

    def test_trace_aliases_normalize_but_raw_paths_and_pid_are_preserved(self):
        entry, metadata, primary = self.sparse_inputs()
        alias = str(primary.parent) + "/./" + primary.name
        first = evidence.validate_loaded_trace(entry, metadata, self.trace([self.loader, alias]), str(self.program), 431, self.depfile)
        second = evidence.validate_loaded_trace(entry, metadata, self.trace([primary, self.loader], pid=999), str(self.program), 999, self.depfile)
        self.assertEqual(first["initialized_objects"], second["initialized_objects"])
        self.assertIn(alias, first["process"]["initialized_paths"])
        self.assertNotEqual(first["process"]["trace_sha256"], second["process"]["trace_sha256"])

    def test_header_only_gpu_inputs_never_require_header_loader_events(self):
        metadata = self.gpu_metadata()
        entry = self.entry("rocthrust", "hip", "benchmark", "hip-rocthrust", [])
        hip = Path(metadata["dependencies"]["hip"]["path"])
        self.links([hip])
        result = evidence.validate_loaded_trace(entry, metadata, self.trace([self.loader, hip]), str(self.program), 431, self.depfile)
        self.assertEqual(result["dependencies"]["rocthrust"]["kind"], "header")
        self.assertEqual(result["dependencies"]["rocprim"]["kind"], "header")
        header = Path(metadata["dependencies"]["rocthrust"]["path"])
        with self.assertRaises(ValueError):
            evidence.validate_loaded_trace(entry, metadata, self.trace([self.loader, hip, header]), str(self.program), 431, self.depfile)

    def test_hash_change_during_read_and_unsafe_inputs_fail_closed(self):
        original = evidence.os.fstat
        call_count = [0]
        def changed(fd):
            value = original(fd)
            call_count[0] += 1
            if call_count[0] == 2:
                return type("Changed", (), {"st_dev": value.st_dev, "st_ino": value.st_ino,
                    "st_size": value.st_size + 1, "st_mtime_ns": value.st_mtime_ns,
                    "st_ctime_ns": value.st_ctime_ns})()
            return value
        with mock.patch.object(evidence.os, "fstat", changed), self.assertRaises(ValueError):
            evidence._file_identity(self.program)
        for path in ("relative", "/tmp/unsafe\nname", "/tmp/lib.so (deleted)"):
            with self.subTest(path=path), self.assertRaises(ValueError): evidence._absolute(path)
        for pid in (0, -1, True):
            with self.subTest(pid=pid), self.assertRaises(ValueError):
                evidence.parse_ld_debug_trace(self.trace([self.loader]), str(self.program), pid)


if __name__ == "__main__":
    unittest.main()
