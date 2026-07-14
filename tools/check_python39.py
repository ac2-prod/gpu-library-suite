#!/usr/bin/env python3
"""Parse and byte-compile every repository Python file as a Python 3.9 audit."""

import ast
import hashlib
import py_compile
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


class CompatibilityVisitor(ast.NodeVisitor):
    """Flag selected stdlib APIs added after Python 3.9."""

    FORBIDDEN_IMPORTS = {"tomllib"}
    FORBIDDEN_IMPORT_MEMBERS = {
        "contextlib": {"chdir"},
        "dataclasses": {"KW_ONLY"},
        "hashlib": {"file_digest"},
        "inspect": {"get_annotations"},
        "itertools": {"batched", "pairwise"},
        "statistics": {"correlation", "linear_regression"},
        "typing": {
            "Concatenate", "LiteralString", "NotRequired", "ParamSpec",
            "ParamSpecArgs", "ParamSpecKwargs", "Required", "Self",
            "TypeAlias", "TypeGuard", "TypeVarTuple", "Unpack",
            "assert_never", "dataclass_transform", "is_typeddict",
            "reveal_type",
        },
    }
    FORBIDDEN_ATTRIBUTES = {
        "batched",
        "chdir",
        "correlation",
        "file_digest",
        "get_annotations",
        "linear_regression",
        "pairwise",
        "walk",
    }
    FORBIDDEN_BUILTINS = {"aiter", "anext"}
    POST39_KEYWORDS = {
        "dataclass": {"kw_only", "match_args", "slots", "weakref_slot"},
        "field": {"kw_only"},
        "glob": {"case_sensitive", "include_hidden", "recurse_symlinks", "root_dir"},
        "rglob": {"case_sensitive", "recurse_symlinks"},
        "read_text": {"newline"},
        "write_text": {"newline"},
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self.errors = []  # type: List[str]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", 1)[0] in self.FORBIDDEN_IMPORTS:
                self.errors.append(
                    "{0}:{1}: Python 3.10+ module {2}".format(
                        self.path, node.lineno, alias.name
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".", 1)[0] in self.FORBIDDEN_IMPORTS:
            self.errors.append(
                "{0}:{1}: Python 3.10+ module {2}".format(
                    self.path, node.lineno, node.module
                )
            )
        if node.module in self.FORBIDDEN_IMPORT_MEMBERS:
            forbidden = self.FORBIDDEN_IMPORT_MEMBERS[node.module]
            for alias in node.names:
                if alias.name in forbidden or alias.name == "*":
                    self.errors.append(
                        "{0}:{1}: audit Python 3.10+ import {2}.{3}".format(
                            self.path, node.lineno, node.module, alias.name
                        )
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            self.errors.append(
                "{0}:{1}: audit Python 3.10+ attribute {2}".format(
                    self.path, node.lineno, node.attr
                )
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function_name = None  # type: Optional[str]
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        if function_name in self.FORBIDDEN_BUILTINS:
            self.errors.append(
                "{0}:{1}: builtin {2} requires Python 3.10".format(
                    self.path, node.lineno, function_name
                )
            )
        if isinstance(node.func, ast.Name) and node.func.id == "zip":
            if any(keyword.arg == "strict" for keyword in node.keywords):
                self.errors.append(
                    "{0}:{1}: zip(strict=...) requires Python 3.10".format(
                        self.path, node.lineno
                    )
                )
        if function_name in self.POST39_KEYWORDS:
            forbidden_keywords = self.POST39_KEYWORDS[function_name]
            for keyword in node.keywords:
                if keyword.arg in forbidden_keywords:
                    self.errors.append(
                        "{0}:{1}: {2}({3}=...) is newer than Python 3.9".format(
                            self.path, node.lineno, function_name, keyword.arg
                        )
                    )
        self.generic_visit(node)


def python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if ".git" not in path.parts and "__pycache__" not in path.parts:
            yield path


def check_file(path: Path, cache_root: Path) -> List[str]:
    errors = []  # type: List[str]
    try:
        source = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        return ["{0}: {1}".format(path, error)]
    try:
        tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
    except SyntaxError as error:
        errors.append("{0}: Python 3.9 syntax error: {1}".format(path, error))
        return errors
    visitor = CompatibilityVisitor(path)
    visitor.visit(tree)
    errors.extend(visitor.errors)
    try:
        compile(source, str(path), "exec", dont_inherit=True)
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        cache_path = cache_root / (digest + ".pyc")
        py_compile.compile(str(path), cfile=str(cache_path), doraise=True)
    except (SyntaxError, py_compile.PyCompileError, OSError) as error:
        errors.append("{0}: byte compilation failed: {1}".format(path, error))
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: check_python39.py REPOSITORY", file=sys.stderr)
        return 2
    root = Path(arguments[0]).resolve()
    cache_root = Path("/tmp/gpu-library-suite-local-build/python39-pyc")
    cache_root.mkdir(parents=True, exist_ok=True)
    files = list(python_files(root))
    errors = []  # type: List[str]
    for path in files:
        errors.extend(check_file(path, cache_root))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Python 3.9 compatibility: {0} files passed".format(len(files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
