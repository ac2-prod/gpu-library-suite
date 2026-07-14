import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

LIBRARY_STEMS = {
    "cufft": (
        "fft_cpu", "fft_gpu", "openacc_cufft", "fft_cpu_bench",
        "fft_gpu_bench", "openacc_cufft_bench",
    ),
    "cublas": (
        "blas_cpu", "blas_gpu", "openacc_cublas", "blas_cpu_bench",
        "blas_gpu_bench", "openacc_cublas_bench",
    ),
    "cusparse": (
        "sparse_cpu", "sparse_gpu", "openacc_cusparse", "sparse_cpu_bench",
        "sparse_gpu_bench", "openacc_cusparse_bench",
    ),
    "cusolver": (
        "solver_cpu", "solver_gpu", "openacc_cusolver", "solver_cpu_bench",
        "solver_gpu_bench", "openacc_cusolver_bench",
    ),
    "curand": (
        "rand_cpu", "rand_gpu", "openacc_curand", "rand_cpu_bench",
        "rand_gpu_bench", "openacc_curand_bench",
    ),
    "thrust": (
        "reduce_cpu", "reduce_gpu", "openacc_thrust", "reduce_cpu_bench",
        "reduce_gpu_bench", "openacc_thrust_bench",
    ),
}


class DocumentationTests(unittest.TestCase):
    def test_every_canonical_name_is_in_root_and_library_readmes(self):
        root_readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        for library, stems in LIBRARY_STEMS.items():
            library_readme = (
                REPOSITORY_ROOT / "nvidia" / "c-cpp" / library / "README.md"
            ).read_text(encoding="utf-8")
            for stem in stems:
                self.assertIn("`{0}`".format(stem), root_readme)
                self.assertIn("`{0}`".format(stem), library_readme)

    def test_user_documentation_records_required_roles_and_boundaries(self):
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("cpu-fftw-threaded", readme)
        self.assertIn("cpu-fftw-serial", readme)
        self.assertIn("Serial CPU baseline", readme)
        self.assertIn("same distribution and output type", readme)
        self.assertIn("RNG algorithms differ", readme)
        self.assertIn("only writer of the node-level raw-result file", readme)
        self.assertIn("starting points, not final production", readme)
        self.assertIn("No real CUDA GPU", readme)

    def test_all_relative_markdown_links_resolve(self):
        link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
        errors = []
        for document in sorted(REPOSITORY_ROOT.rglob("*.md")):
            if ".git" in document.parts:
                continue
            text = document.read_text(encoding="utf-8")
            for raw_target in link_re.findall(text):
                target = raw_target.strip().split("#", 1)[0]
                if (
                    target == ""
                    or "://" in target
                    or target.startswith("mailto:")
                ):
                    continue
                resolved = (document.parent / target).resolve()
                if not resolved.exists():
                    errors.append(
                        "{0}: unresolved link {1}".format(
                            document.relative_to(REPOSITORY_ROOT), raw_target
                        )
                    )
        self.assertEqual(errors, [])

    def test_implemented_documents_have_no_stale_preimplementation_language(self):
        values = {
            "docs/PEGASUS_EXECUTION.md": (
                "将来の`jobs/pegasus/pegasus.json.example`",
                "将来のtemplate",
                "将来の`jobs/pegasus/`成果物",
            ),
            "docs/RESULT_SCHEMA.md": ("same future Python utility",),
            "docs/IMPLEMENTATION_PLAN.md": (
                "no initial implementation",
                "Portable configuration requests one CPU thread",
                "All 30 target/executable names",
            ),
        }
        for relative, stale_values in values.items():
            text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
            for stale in stale_values:
                self.assertNotIn(stale, text)

    def test_no_versioned_canonical_files_or_empty_future_trees(self):
        versioned = re.compile(
            r"(?:benchmark-v[12]|pilot-v1|run_suite_v1|-v[12]\.json|"
            r"-final\.json|-latest\.json)$"
        )
        bad_paths = [
            str(path.relative_to(REPOSITORY_ROOT))
            for path in REPOSITORY_ROOT.rglob("*")
            if versioned.search(path.name)
        ]
        self.assertEqual(bad_paths, [])
        for relative in (
            "nvidia/fortran", "amd", "common/fortran", "jobs/furo"
        ):
            self.assertFalse((REPOSITORY_ROOT / relative).exists())

    def test_portable_files_have_no_embedded_gpu_architecture(self):
        architecture_re = re.compile(r"(?:sm_[0-9]+|cc[0-9]{2,})")
        bad = []
        roots = (
            REPOSITORY_ROOT / "CMakeLists.txt",
            REPOSITORY_ROOT / "cmake",
            REPOSITORY_ROOT / "common",
            REPOSITORY_ROOT / "nvidia",
            REPOSITORY_ROOT / "configs",
            REPOSITORY_ROOT / "tools",
        )
        paths = []
        for root in roots:
            paths.extend([root] if root.is_file() else root.rglob("*"))
        for path in paths:
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeError:
                    continue
                if architecture_re.search(text):
                    bad.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(bad, [])

    def test_job_programs_do_not_operate_remote_or_scheduler_commands(self):
        forbidden = re.compile(
            r"(^|[^A-Za-z0-9_])(?:qsub|qstat|qdel|qlogin|sstat|pegasusinfo|"
            r"rbudgetcheck|ssh|scp|sftp|rsync)([^A-Za-z0-9_]|$)"
        )
        bad = []
        for suffix in ("*.py", "*.sh", "*.in"):
            for path in (REPOSITORY_ROOT / "jobs" / "pegasus").rglob(suffix):
                text = path.read_text(encoding="utf-8")
                if forbidden.search(text):
                    bad.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
