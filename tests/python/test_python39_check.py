import ast
import tempfile
import unittest
from pathlib import Path

from check_python39 import CompatibilityVisitor, python_files


def compatibility_errors(source):
    tree = ast.parse(source, feature_version=(3, 9))
    visitor = CompatibilityVisitor(Path("fixture.py"))
    visitor.visit(tree)
    return visitor.errors


class Python39CompatibilityAuditTests(unittest.TestCase):
    def test_post39_imports_and_builtins_are_rejected(self):
        errors = compatibility_errors(
            "from itertools import pairwise\n"
            "from typing import Self\n"
            "value = anext(iterator)\n"
        )
        self.assertEqual(len(errors), 3)

    def test_post39_keyword_extensions_are_rejected(self):
        errors = compatibility_errors(
            "from pathlib import Path\n"
            "Path('x').write_text('x', newline='\\n')\n"
            "zip([1], strict=True)\n"
        )
        self.assertEqual(len(errors), 2)

    def test_python39_apis_are_accepted(self):
        errors = compatibility_errors(
            "from pathlib import Path\n"
            "text = Path('x').read_text(encoding='utf-8')\n"
            "value = 'prefix'.removeprefix('pre')\n"
        )
        self.assertEqual(errors, [])

    def test_inventory_excludes_ignored_validation_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "tools" / "source.py"
            ignored = root / "manual-validation" / "raw" / "ignored.py"
            source.parent.mkdir()
            ignored.parent.mkdir(parents=True)
            source.write_text("value = 1\n", encoding="utf-8")
            ignored.write_text("from typing import Self\n", encoding="utf-8")
            self.assertEqual(list(python_files(root)), [source])


if __name__ == "__main__":
    unittest.main()
