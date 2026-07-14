import hashlib
import os
import re
import unittest
from pathlib import Path

from gpu_suite.strict_json import dump_bytes, loads


class BuildMetadataTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("GPU_SUITE_BUILD_METADATA_FILE"),
        "CMake-generated metadata path is not set",
    )
    def test_generated_metadata_is_deterministic_and_hashed(self):
        metadata_path = Path(os.environ["GPU_SUITE_BUILD_METADATA_FILE"])
        header_path = Path(os.environ["GPU_SUITE_BUILD_METADATA_HEADER"])
        content = metadata_path.read_bytes()
        metadata = loads(content)
        self.assertEqual(content, dump_bytes(metadata))
        self.assertIn(metadata["build_profile"], {"cpu-cuda", "openacc"})
        self.assertIn("architectures", metadata["cuda"])
        self.assertIn("toolkit_root", metadata["cuda"])
        self.assertIn("toolkit_version", metadata["cuda"])
        self.assertIn("cuda_home", metadata["nvhpc"])
        self.assertIn("gpu_target", metadata["nvhpc"])
        self.assertEqual(
            set(metadata["openacc"]),
            {
                "compile_flags",
                "link_flags",
                "thrust_cuda_interop_compile_flags",
                "thrust_cuda_interop_link_flags",
            },
        )
        digest = hashlib.sha256(content).hexdigest()
        header = header_path.read_text(encoding="utf-8")
        match = re.search(
            r'GPU_SUITE_BUILD_METADATA_SHA256\s+\\\s*\n\s*"([0-9a-f]{64})"',
            header,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), digest)


if __name__ == "__main__":
    unittest.main()
