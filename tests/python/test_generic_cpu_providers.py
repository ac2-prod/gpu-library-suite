"""Host archive fixtures for explicit generic CPU providers and dependency identity."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gpu_suite.runner import RunnerError, _validate_amd_dependencies

HEADERS = ROOT / "tests/fixtures/cpu_libs/include"
GENERATOR = "Ninja" if shutil.which("ninja") else "Unix Makefiles"
PROVIDER = r'''
#include "cblas.h"
#include "lapacke.h"
#include <stdlib.h>
extern int fixture_first(void);
void cblas_dgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb,
    int m, int n, int k, double alpha, const double *a, int lda,
    const double *b, int ldb, double beta, double *c, int ldc) {
  if (fixture_first() != 7) abort();
  for (int j=0; j<n; ++j) for (int i=0; i<m; ++i) {
    double sum=0;
    for (int x=0; x<k; ++x) sum += a[i+x*lda]*b[x+j*ldb];
    c[i+j*ldc] = alpha*sum + beta*c[i+j*ldc];
  }
}
lapack_int LAPACKE_dgesv(int layout, lapack_int n, lapack_int nrhs,
    double *a, lapack_int lda, lapack_int *p, double *b, lapack_int ldb) {
  return fixture_first()-7;
}
lapack_int LAPACKE_dgetrf(int layout, lapack_int m, lapack_int n,
    double *a, lapack_int lda, lapack_int *p) { return fixture_first()-7; }
lapack_int LAPACKE_dgetrs(int layout, char trans, lapack_int n, lapack_int nrhs,
    const double *a, lapack_int lda, const lapack_int *p, double *b,
    lapack_int ldb) { return fixture_first()-7; }
'''
CONSUMER = r'''
#include "cblas.h"
#include "lapacke.h"
int main(void) {
  double a=2, b=3, c=0; lapack_int p=0;
  cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,1,1,1,1,&a,1,&b,1,0,&c,1);
  return c != 6 || LAPACKE_dgesv(LAPACK_COL_MAJOR,1,1,&a,1,&p,&b,1) ||
    LAPACKE_dgetrf(LAPACK_COL_MAJOR,1,1,&a,1,&p) ||
    LAPACKE_dgetrs(LAPACK_COL_MAJOR,'N',1,1,&a,1,&p,&b,1);
}
'''
CMAKE_SOURCE = r'''
cmake_minimum_required(VERSION 3.20)
project(generic_provider_fixture C)
set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
list(APPEND CMAKE_MODULE_PATH "@MODULES@")
include(GpuSuiteCpuDependencies)
include(GpuSuiteBuildMetadata)
include(GpuSuiteAmdMetadata)
gpu_suite_find_cblas()
gpu_suite_find_lapacke()
foreach(name CBLAS LAPACKE)
  _gpu_suite_escape_json_string("${GPU_SUITE_${name}_REASON}" reason_${name})
endforeach()
file(WRITE "${CMAKE_BINARY_DIR}/status.json"
  "{\"cblas\":\"${GPU_SUITE_CBLAS_FOUND}\",\"lapacke\":\"${GPU_SUITE_LAPACKE_FOUND}\",\"cblas_reason\":\"${reason_CBLAS}\",\"lapacke_reason\":\"${reason_LAPACKE}\"}")
if(GPU_SUITE_CBLAS_FOUND AND GPU_SUITE_LAPACKE_FOUND)
  add_executable(provider_consumer main.c)
  target_link_libraries(provider_consumer PRIVATE GpuSuite::CBLAS GpuSuite::LAPACKE)
endif()
gpu_suite_prepare_amd_metadata()
file(WRITE "${CMAKE_BINARY_DIR}/metadata.json"
  "{\"vendor\":\"amd\",\"dependencies\":${GPU_SUITE_META_DEPENDENCIES_JSON},\"amd_build_inputs\":${GPU_SUITE_META_AMD_INPUTS_JSON}}")
'''


def temporary_root():
    allowed = Path("/tmp/gpu-library-suite-local-build")
    allowed.mkdir(parents=True, exist_ok=True)
    requested = Path(tempfile.gettempdir()).resolve()
    return requested if requested == allowed.resolve() or allowed.resolve() in requested.parents else allowed


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc") and shutil.which("ar"),
                     "CMake, host C compiler and archive tool are required")
class GenericCpuProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory(prefix="generic-provider-libs-", dir=temporary_root())
        cls.addClassCleanup(temporary.cleanup)
        cls.fixture = Path(temporary.name)
        cls.environment = os.environ.copy()
        for name in ("CC", "CFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH",
                     "LIBRARY_PATH", "LD_LIBRARY_PATH", "MKLROOT", "CMAKE_PREFIX_PATH",
                     "CMAKE_LIBRARY_PATH", "CMAKE_INCLUDE_PATH"):
            cls.environment.pop(name, None)
        cls.environment["TMPDIR"] = str(cls.fixture)
        sources = {
            "chosen_numeric": PROVIDER,
            "first": "extern int fixture_last(void); int fixture_first(void) { return fixture_last(); }\n",
            "last": "int fixture_last(void) { return 7; }\n",
            "missing_api": "int fixture_unrelated(void) { return 0; }\n",
            "standalone": PROVIDER + "\nint fixture_first(void) { return 7; }\n",
        }
        for name, source in sources.items():
            path = cls.fixture / (name + ".c")
            path.write_text(source)
            for argv in (["cc", "-std=c17", "-I", str(HEADERS), "-c", str(path),
                          "-o", str(path.with_suffix(".o"))],
                         ["ar", "rcs", str(cls.fixture / ("lib" + name + ".a")),
                          str(path.with_suffix(".o"))]):
                result = subprocess.run(argv, env=cls.environment, capture_output=True,
                                        text=True, check=False)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="generic-provider-", dir=temporary_root())
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.source = self.work / "source"
        self.source.mkdir()
        (self.source / "CMakeLists.txt").write_text(
            CMAKE_SOURCE.replace("@MODULES@", str(ROOT / "cmake")))
        (self.source / "main.c").write_text(CONSUMER)
        self.primary = self.work / "libchosen_numeric.a"
        self.first = self.work / "libfirst.a"
        self.last = self.work / "liblast.a"
        for path in (self.primary, self.first, self.last):
            shutil.copyfile(self.fixture / path.name, path)
        self.environment = dict(self.environment, TMPDIR=str(self.work))

    def command(self, argv):
        result = subprocess.run([str(item) for item in argv], env=self.environment,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def configure(self, name="build", primary=None, extras=None, provider="GENERIC"):
        build = self.work / name
        primary = self.primary if primary is None else primary
        extras = [self.first, self.last] if extras is None else extras
        cblas = "GENERIC_CBLAS" if provider == "GENERIC" else provider
        lapacke = "GENERIC_LAPACKE" if provider == "GENERIC" else provider
        cblas_prefix = "GENERIC" if provider == "GENERIC" else provider
        lapacke_prefix = cblas_prefix
        self.command(["cmake", "-S", self.source, "-B", build, "-G", GENERATOR,
                      "-DGPU_SUITE_CPU_BLAS_BACKEND=" + cblas,
                      "-DGPU_SUITE_CPU_LAPACK_BACKEND=" + lapacke,
                      "-DGPU_SUITE_" + cblas_prefix + "_CBLAS_INCLUDE_DIR=" + str(HEADERS),
                      "-DGPU_SUITE_" + cblas_prefix + "_CBLAS_LIBRARY=" + str(primary),
                      "-DGPU_SUITE_" + lapacke_prefix + "_LAPACKE_INCLUDE_DIR=" + str(HEADERS),
                      "-DGPU_SUITE_" + lapacke_prefix + "_LAPACKE_LIBRARY=" + str(primary),
                      "-DGPU_SUITE_GENERIC_CBLAS_EXTRA_LIBRARIES=" + ";".join(map(str, extras)),
                      "-DGPU_SUITE_GENERIC_LAPACKE_EXTRA_LIBRARIES=" + ";".join(map(str, extras))])
        return build, json.loads((build / "status.json").read_text())

    def assert_disabled(self, build, status, reason):
        for name in ("cblas", "lapacke"):
            self.assertEqual(status[name], "FALSE")
            self.assertIn(reason, status[name + "_reason"])
        metadata = json.loads((build / "metadata.json").read_text())
        self.assertEqual(metadata["dependencies"], {})
        output = self.command(["cmake", "--build", build, "--target", "help"])
        self.assertNotIn("provider_consumer", output.stdout)

    def test_explicit_generic_files_link_and_record_dependency_hashes_in_order(self):
        build, status = self.configure()
        self.assertEqual((status["cblas"], status["lapacke"]), ("TRUE", "TRUE"))
        self.command(["cmake", "--build", build, "--target", "provider_consumer"])
        self.command([build / "provider_consumer"])
        metadata = json.loads((build / "metadata.json").read_text())
        for prefix, target in (("cblas", "CBLAS"), ("lapacke", "LAPACKE")):
            for index, path in enumerate((self.primary, self.first, self.last)):
                key = prefix if index == 0 else prefix + "_link_" + str(index)
                self.assertEqual(metadata["dependencies"][key], {
                    "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            link_inputs = metadata["amd_build_inputs"][target + "_INTERFACE_LINK_LIBRARIES"].split(";")
            self.assertEqual(link_inputs[:3], list(map(str, (self.primary, self.first, self.last))))

    def test_missing_extra_dependency_is_not_silently_replaced(self):
        build, status = self.configure(extras=[self.first])
        self.assert_disabled(build, status, "did not compile and link")

    def test_missing_api_does_not_enable_generic_targets(self):
        build, status = self.configure(primary=self.fixture / "libmissing_api.a")
        self.assert_disabled(build, status, "did not compile and link")

    def test_generic_inputs_must_be_existing_absolute_files(self):
        for name, primary, extras in (
                ("relative-main", "libchosen_numeric.a", []),
                ("missing-extra", self.primary, [self.work / "missing.a"]),
                ("relative-extra", self.primary, ["libfirst.a"]),
                ("empty-extra", self.primary, [self.first, "", self.last]),
                ("directory-extra", self.primary, [self.work]),
                ("flag-extra", self.primary, ["-lm"])):
            with self.subTest(name=name):
                build, status = self.configure(name=name, primary=primary, extras=extras)
                self.assert_disabled(build, status, "existing absolute files")

    def test_named_provider_filename_checks_remain_strict(self):
        for provider, filename in (("ONEMKL", "libmkl_rt.a"), ("OPENBLAS", "libopenblas.a")):
            with self.subTest(provider=provider):
                build, status = self.configure(name=provider + "-wrong", provider=provider)
                self.assert_disabled(build, status, "different provider")
                named_library = self.work / filename
                shutil.copyfile(self.fixture / "libstandalone.a", named_library)
                build, status = self.configure(name=provider + "-correct", provider=provider,
                                               primary=named_library)
                self.assertEqual((status["cblas"], status["lapacke"]), ("TRUE", "TRUE"))
                self.command(["cmake", "--build", build, "--target", "provider_consumer"])
                self.command([build / "provider_consumer"])

    def test_runner_rejects_changed_extra_dependency(self):
        build, status = self.configure()
        self.assertEqual(status["cblas"], "TRUE")
        metadata = json.loads((build / "metadata.json").read_text())
        entry = {"library": "rocblas", "implementation": "cpu",
                 "supported_cpu_backends": ["cpu-generic-cblas"]}
        _validate_amd_dependencies(entry, metadata)
        self.last.write_bytes(self.last.read_bytes() + b"\nchanged extra dependency\n")
        with self.assertRaisesRegex(RunnerError, "resolved dependency changed since configure: cblas_link_2"):
            _validate_amd_dependencies(entry, metadata)


if __name__ == "__main__":
    unittest.main()
