import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SOURCE_MARKDOWN_ROOTS = (
    "amd", "common", "configs", "docs", "jobs", "nvidia", "tests", "tools",
)
SOURCE_TREE_ROOTS = SOURCE_MARKDOWN_ROOTS + ("cmake",)


def _git_markdown_documents(root: Path) -> Optional[List[Path]]:
    if not (root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            [
                "git", "-C", str(root), "ls-files", "-z", "--cached",
                "--others", "--exclude-standard", "--", "*.md",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        relative_paths = completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    return [
        root / relative
        for relative in relative_paths.split("\0") if relative
        if (root / relative).is_file()
    ]


def _is_generated_path(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in {"build", "manual-validation", "results"}:
        return True
    if any(part in {".git", "CMakeFiles", "_deps", "test-fixtures"}
           for part in parts):
        return True
    if parts[:3] == ("jobs", "pegasus", "generated"):
        return True
    for parent in path.parents:
        if parent == root:
            break
        if ((parent / "CMakeCache.txt").is_file()
                or (parent / "CMakeFiles").is_dir()):
            return True
    return False


def _fallback_markdown_documents(root: Path) -> List[Path]:
    candidates = list(root.glob("*.md"))
    for relative in SOURCE_MARKDOWN_ROOTS:
        source_root = root / relative
        if source_root.is_dir():
            candidates.extend(source_root.rglob("*.md"))
    return [
        path for path in candidates
        if path.is_file() and not _is_generated_path(root, path)
    ]


def _source_tree_paths(root: Path) -> List[Path]:
    candidates = [path for path in root.iterdir() if path.is_file()]
    for relative in SOURCE_TREE_ROOTS:
        source_root = root / relative
        if source_root.is_dir():
            candidates.extend(source_root.rglob("*"))
    return [
        path for path in candidates
        if not _is_generated_path(root, path)
    ]


def _source_markdown_documents(root: Path) -> List[Path]:
    git_documents = _git_markdown_documents(root)
    return (
        git_documents
        if git_documents is not None
        else _fallback_markdown_documents(root)
    )

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
    def test_library_readmes_have_the_required_common_sections(self):
        headings = (
            "## Canonical sources and targets",
            "## Teaching default",
            "## Dependencies",
            "## Direct compile",
            "## CMake configure and build",
            "## Benchmark CLI",
            "## Timing scopes",
            "## Verification",
            "## CPU backend and role",
            "## OpenACC notes",
            "## Known limitations and local validation",
        )
        for library in LIBRARY_STEMS:
            path = (
                REPOSITORY_ROOT / "nvidia" / "c-cpp" / library / "README.md"
            )
            text = path.read_text(encoding="utf-8")
            with self.subTest(library=library):
                for heading in headings:
                    self.assertIn(heading, text)
                self.assertIn("`compute`", text)
                self.assertIn("`end-to-end`", text)
                self.assertIn("locally unverified", text)

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
        for document in sorted(_source_markdown_documents(REPOSITORY_ROOT)):
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

    def test_non_git_markdown_inventory_excludes_generated_trees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            included = (root / "README.md", root / "docs" / "guide.md")
            excluded = (
                root / "manual-validation" / "run" / "README.md",
                root / "results" / "README.md",
                root / "tests" / "test-fixtures" / "source" / "README.md",
                root / "jobs" / "pegasus" / "generated" / "README.md",
                root / "docs" / "cmake-build" / "README.md",
            )
            for path in included + excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[missing](does-not-exist.md)\n", encoding="utf-8")
            (root / "docs" / "cmake-build" / "CMakeCache.txt").write_text(
                "generated build tree\n", encoding="utf-8"
            )
            documents = {
                path.relative_to(root)
                for path in _source_markdown_documents(root)
            }
        self.assertEqual(documents, {Path("README.md"), Path("docs/guide.md")})

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
            for path in _source_tree_paths(REPOSITORY_ROOT)
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
