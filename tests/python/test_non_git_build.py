import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from check_non_git_build import (
        assert_not_git_worktree,
        isolated_git_environment,
    )
except ModuleNotFoundError:
    from tests.python.check_non_git_build import (
        assert_not_git_worktree,
        isolated_git_environment,
    )


ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("GPU_SUITE_CMAKE_COMMAND")


class NonGitBuildIsolationTests(unittest.TestCase):
    def configure_metadata_fixture(self, parent):
        source = parent / "source"
        build = parent / "build"
        source.mkdir()
        module = (ROOT / "cmake" / "GpuSuiteBuildMetadata.cmake").as_posix()
        (source / "CMakeLists.txt").write_text(
            "\n".join([
                "cmake_minimum_required(VERSION 3.20)",
                "project(non_git_metadata LANGUAGES NONE)",
                'include("{0}")'.format(module),
                "set(GPU_SUITE_BUILD_OPENACC OFF)",
                "gpu_suite_generate_build_metadata()",
                "",
            ]),
            encoding="utf-8",
        )
        environment = isolated_git_environment(source)
        assert_not_git_worktree(source, environment)
        completed = subprocess.run(
            [CMAKE, "-S", str(source), "-B", str(build)],
            cwd=str(parent),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        metadata = json.loads(
            (build / "build-metadata.json").read_text(encoding="utf-8")
        )
        return (
            metadata.get("git_metadata_available"),
            metadata.get("git_commit"),
            metadata.get("git_dirty"),
        )

    @unittest.skipUnless(CMAKE, "GPU_SUITE_CMAKE_COMMAND is not set")
    def test_inside_and_outside_repository_are_equally_non_git(self):
        with tempfile.TemporaryDirectory(
            prefix=".gpu-suite-non-git-inside-", dir=str(ROOT)
        ) as inside_directory:
            inside = self.configure_metadata_fixture(Path(inside_directory))
        with tempfile.TemporaryDirectory(
            prefix="gpu-suite-non-git-outside-"
        ) as outside_directory:
            outside = self.configure_metadata_fixture(Path(outside_directory))
        self.assertEqual(inside, (False, None, None))
        self.assertEqual(outside, (False, None, None))
        self.assertEqual(inside, outside)


if __name__ == "__main__":
    unittest.main()
