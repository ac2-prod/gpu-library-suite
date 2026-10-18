#!/usr/bin/env python3
"""Bounded, static AOCL module inventory; never evaluate a module or shell file.

The results describe declarations and file candidates, not an effective module
environment. Dynamic configuration remains explicitly unresolved.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import shlex
import sys


MODULES = ("aocl/5.1.0/gcc14.2.1", "aocl/5.1.0/aocc5.0.0")
MAX_FILES = 64
MAX_FILE_BYTES = 131072
MAX_TOTAL_BYTES = 1048576
MAX_ROOTS = 32
MAX_CONFIGS = 64
MAX_DEPTH = 4
MAX_ENTRIES = 512
MAX_CANDIDATE_DIRS = 64
CONFIGS = (
    "/etc/profile.d/modules.sh", "/etc/profile.d/lmod.sh",
    "/etc/environment-modules/modulespath", "/etc/environment-modules/initrc",
    "/usr/share/Modules/init/.modulespath", "/usr/share/Modules/init/modulespath",
    "/usr/share/Modules/init/bash", "/etc/lmod/.modulespath",
    "/etc/lmod/modulespath", "/etc/lmod/lmodrc.lua",
    "/usr/share/lmod/lmod/init/.modulespath", "/usr/share/lmod/lmod/init/bash",
)
PATH_VARIABLES = {
    "MODULEPATH", "MODULESHOME", "LMOD_CMD", "AOCL_ROOT", "AOCL_HOME",
    "FFTW_ROOT", "FFTW3_ROOT", "FFTW3F_ROOT", "FFTW3f_ROOT",
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
    "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "CMAKE_INCLUDE_PATH",
    "CMAKE_LIBRARY_PATH", "PKG_CONFIG_PATH", "PATH",
}
LIBRARY = re.compile(r"libfftw3f(?:_threads|_omp)?\.(?:a|so(?:\.[0-9][A-Za-z0-9.]*)?)\Z")


def without_comment(line, lua=False):
    """Strip shell/Tcl/Lua comments without interpreting quoted contents."""
    quote = None
    for index, character in enumerate(line):
        if quote:
            if character == quote:
                quote = None
        elif character in ("'", '"'):
            quote = character
        elif character == "#" or (lua and line[index:index + 2] == "--"):
            if index == 0 or line[index - 1].isspace():
                return line[:index].strip()
    return line.strip()


def literal(expression, variables):
    """Resolve only literals, literal variables and one-level path joins."""
    value = expression.strip()
    if not value or any(token in value for token in (";", "`", "$(", "\n")):
        return None
    if value.startswith("{") and "$" in value:
        return None  # Tcl braces suppress expansion; do not reinterpret them.
    if re.search(r"\s\.\.\s|[\"']\s*\.\.|\.\.\s*[\"']", value):
        return None  # Lua concatenation is deliberately outside this parser.
    joined = re.fullmatch(r"pathJoin\((.*)\)", value)
    tcl_join = re.fullmatch(r"\[file\s+join\s+(.*)\]", value)
    if joined or tcl_join:
        try:
            if joined:
                lexer = shlex.shlex(joined.group(1), posix=False)
                lexer.whitespace = ","
                lexer.whitespace_split = True
                pieces = list(lexer)
            else:
                pieces = shlex.split(tcl_join.group(1), posix=False)
        except ValueError:
            return None
        if not 1 < len(pieces) <= 8:
            return None
        resolved = [literal(piece, variables) for piece in pieces]
        if any(piece is None for piece in resolved):
            return None
        return str(Path(resolved[0]).joinpath(*resolved[1:]))
    if any(token in value for token in ("[", "]", "(", ")")):
        return None
    if len(value) >= 2 and (value[0], value[-1]) in (("'", "'"), ('"', '"'), ("{", "}")):
        value = value[1:-1]
    elif re.fullmatch(r"[A-Za-z_]\w*", value) and value in variables:
        return variables[value]
    missing = False

    def replace(match):
        nonlocal missing
        replacement = variables.get(match.group(1) or match.group(2))
        if replacement is None:
            missing = True
            return ""
        return replacement

    value = re.sub(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)", replace, value)
    if missing or "$" in value or "\\" in value:
        return None
    return value


class Inventory:
    def __init__(self, system_root, env):
        self.system_root = Path(system_root).resolve()
        self.env = dict(env)
        self.records = []
        self.roots = {}
        self.pending = deque()
        self.seen_configs = set()
        self.read_files = 0
        self.read_bytes = 0
        self.inspected_dirs = set()
        self.checked_files = set()

    def emit(self, kind, **fields):
        self.records.append(dict(kind=kind, **fields))

    def physical(self, path):
        logical = Path(path)
        if not logical.is_absolute() or ".." in logical.parts:
            self.emit("unresolved", path=str(path), reason="nonabsolute_or_parent_path")
            return None
        physical = self.system_root / str(logical).lstrip("/")
        try:
            physical.resolve().relative_to(self.system_root)
        except (ValueError, OSError, RuntimeError):
            self.emit("unresolved", path=str(path), reason="path_escapes_inventory_root")
            return None
        return physical

    def read(self, path, purpose):
        physical = self.physical(path)
        if physical is None:
            return None
        if self.read_files >= MAX_FILES:
            self.emit("limit", path=path, reason="files", limit=MAX_FILES)
            return None
        try:
            if not physical.is_file():
                self.emit("missing_file", path=path, purpose=purpose)
                return None
            size = physical.stat().st_size
            if size > MAX_FILE_BYTES or self.read_bytes + size > MAX_TOTAL_BYTES:
                self.emit("limit", path=path, reason="bytes", size=size)
                return None
            with physical.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES or self.read_bytes + len(data) > MAX_TOTAL_BYTES:
                self.emit("limit", path=path, reason="file_grew")
                return None
            self.read_files += 1
            self.read_bytes += len(data)
            return data.decode("utf-8")
        except (OSError, UnicodeError) as error:
            self.emit("unresolved", path=path, reason="unreadable", error=type(error).__name__)
            return None

    def names(self, path):
        physical = self.physical(path)
        if physical is None:
            return []
        names = []
        try:
            with os.scandir(physical) as entries:
                for index, entry in enumerate(entries):
                    if index >= MAX_ENTRIES:
                        self.emit("limit", path=path, reason="directory_entries", limit=MAX_ENTRIES)
                        break
                    names.append(entry.name)
        except OSError as error:
            self.emit("unresolved", path=path, reason="directory_unavailable", error=type(error).__name__)
        return sorted(names)

    def add_root(self, path, source, line=0):
        if not path.startswith("/"):
            self.emit("unresolved", expression=path, source=source, line=line, reason="module_root_not_absolute")
            return
        if path in self.roots:
            return
        if len(self.roots) >= MAX_ROOTS:
            self.emit("limit", reason="module_roots", limit=MAX_ROOTS)
            return
        if self.physical(path) is None:
            return
        self.roots[path] = (source, line)
        self.emit("module_root", path=path, source=source, line=line, status="static_candidate")

    def queue(self, path, depth, source, line=0):
        if depth > MAX_DEPTH:
            self.emit("limit", path=path, reason="config_depth", limit=MAX_DEPTH)
        elif path not in self.seen_configs and len(self.seen_configs) >= MAX_CONFIGS:
            self.emit("limit", path=path, reason="config_candidates", limit=MAX_CONFIGS)
        elif path not in self.seen_configs:
            self.seen_configs.add(path)
            self.pending.append((path, depth, source, line))

    def path_values(self, expression, variables, source, line):
        value = literal(expression, variables)
        if value is None:
            self.emit("unresolved", source=source, line=line, expression=expression, reason="dynamic_path_expression")
            # Retain explicit colon-list components without inventing the rest.
            if ":" not in expression or any(token in expression for token in ("$(", "`", "[", "(", ";")):
                return []
            parts = expression.strip('\"\'').split(":")
            value = ":".join(part for part in parts if part.startswith("/") and "$" not in part)
        paths = value.split(":")
        result = []
        for path in paths:
            if path.startswith("/") and self.physical(path) is not None:
                result.append(path)
            elif path:
                self.emit("unresolved", source=source, line=line, expression=path, reason="path_not_absolute")
        return result

    def config(self, path, depth, source, source_line):
        data = self.read(path, "module_configuration")
        if data is None:
            return
        self.emit("configuration", path=path, source=source, line=source_line, status="static_read_only")
        variables = {key: value for key, value in self.env.items() if key in PATH_VARIABLES}
        for number, raw in enumerate(data.splitlines(), 1):
            text = without_comment(raw, lua=path.endswith(".lua"))
            if not text or text.startswith("#"):
                continue
            assignment = re.fullmatch(r"(?:export\s+)?([A-Za-z_]\w*)\s*=\s*(.*?)\s*;?", text)
            if not assignment:
                assignment = re.fullmatch(r"set\s+([A-Za-z_]\w*)\s+(.+)", text)
            if not assignment:
                assignment = re.fullmatch(r"local\s+([A-Za-z_]\w*)\s*=\s*(.+)", text)
            if assignment:
                name, expression = assignment.groups()
                value = literal(expression, variables)
                if name == "MODULEPATH":
                    for root in self.path_values(expression, variables, path, number):
                        self.add_root(root, path, number)
                elif name in ("MODULESHOME", "LMOD_CMD") and value and value.startswith("/"):
                    self.module_home(name, value, path, number, depth + 1)
                variables[name] = value
                continue
            declaration = re.fullmatch(r"(?:setenv|prepend-path|append-path)\s+MODULEPATH\s+(.+)", text)
            if not declaration:
                declaration = re.fullmatch(r'(?:setenv|prepend_path|append_path)\s*\(\s*[\'"]MODULEPATH[\'"]\s*,\s*(.*)\)\s*', text)
            if declaration:
                for root in self.path_values(declaration.group(1), variables, path, number):
                    self.add_root(root, path, number)
                continue
            reference = re.fullmatch(r"(?:source|\.)\s+(.+)", text)
            if reference:
                for filename in self.path_values(reference.group(1), variables, path, number):
                    self.queue(filename, depth + 1, path, number)
                continue
            use = re.fullmatch(r"module\s+use\s+(?:--append\s+|-a\s+)?(.+)", text)
            if use:
                for root in self.path_values(use.group(1), variables, path, number):
                    self.add_root(root, path, number)
                continue
            if Path(path).name in (".modulespath", "modulespath", "modulepath"):
                for root in self.path_values(text, variables, path, number):
                    self.add_root(root, path, number)
            elif re.search(r"MODULEPATH|module\s+(?:load|use)|LMOD|MODULESHOME|\bsource\b", text):
                self.emit("unresolved", source=path, line=number, expression=text,
                          reason="uninterpreted_module_configuration")

    def module_home(self, name, value, source, line, depth=0):
        home = Path(value) if name == "MODULESHOME" else Path(value).parent.parent
        for suffix in ("init/.modulespath", "init/modulespath", "init/bash", "init/profile"):
            self.queue(str(home / suffix), depth, source, line)

    def candidate_file(self, path, role, source, line):
        if path in self.checked_files:
            return
        self.checked_files.add(path)
        physical = self.physical(path)
        if physical is None:
            return
        try:
            exists = physical.is_file()
        except OSError:
            exists = False
        self.emit("candidate_file" if exists else "missing_file", path=path,
                  role=role, source=source, line=line, status="static_candidate")

    def directory(self, directory, role, source, line):
        key = (directory, role)
        if key in self.inspected_dirs:
            return
        if len(self.inspected_dirs) >= MAX_CANDIDATE_DIRS:
            self.emit("limit", reason="candidate_directories", limit=MAX_CANDIDATE_DIRS)
            return
        self.inspected_dirs.add(key)
        if role == "include":
            self.candidate_file(str(Path(directory) / "fftw3.h"), "header", source, line)
        elif role == "pkgconfig":
            self.candidate_file(str(Path(directory) / "fftw3f.pc"), "pkgconfig", source, line)
        elif role == "library":
            matches = [name for name in self.names(directory) if LIBRARY.fullmatch(name)]
            for name in matches:
                runtime = ".so." in name
                self.candidate_file(str(Path(directory) / name),
                                    "versioned_runtime" if runtime else "link_library", source, line)
            if not matches:
                self.emit("missing_file", path=directory, role="fftw_libraries", source=source, line=line)

    def inspect_path(self, variable, value, source, line):
        if variable == "MODULEPATH":
            self.add_root(value, source, line)
        elif variable in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CMAKE_INCLUDE_PATH"):
            self.directory(value, "include", source, line)
        elif variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH", "CMAKE_LIBRARY_PATH"):
            self.directory(value, "library", source, line)
        elif variable == "PKG_CONFIG_PATH":
            self.directory(value, "pkgconfig", source, line)
        elif variable in ("AOCL_ROOT", "AOCL_HOME", "FFTW_ROOT", "FFTW3_ROOT", "FFTW3F_ROOT", "FFTW3f_ROOT", "CMAKE_PREFIX_PATH"):
            self.directory(str(Path(value) / "include"), "include", source, line)
            for lib in ("lib", "lib64"):
                self.directory(str(Path(value) / lib), "library", source, line)
                self.directory(str(Path(value) / lib / "pkgconfig"), "pkgconfig", source, line)

    def module_file(self, path, module):
        data = self.read(path, "aocl_modulefile")
        if data is None:
            return
        conditional = bool(re.search(r"(?m)^\s*(?:if\b|elseif\b|else\b|for\b|foreach\b|while\b)", data))
        self.emit("modulefile", path=path, module=module, status="static_candidate", conditional=conditional)
        variables = {}
        for number, raw in enumerate(data.splitlines(), 1):
            text = without_comment(raw, lua=path.endswith(".lua"))
            if not text or text.startswith(("#", "--")):
                continue
            assignment = re.fullmatch(r"set\s+([A-Za-z_]\w*)\s+(.+)", text)
            if not assignment:
                assignment = re.fullmatch(r"(?:local\s+)?([A-Za-z_]\w*)\s*=\s*(.+)", text)
            if assignment:
                name, expression = assignment.groups()
                value = literal(expression, variables)
                if name in variables and variables[name] != value:
                    value = None
                    self.emit("unresolved", source=path, line=number, expression=text, reason="variable_reassigned")
                variables[name] = value
                if value is None:
                    self.emit("unresolved", source=path, line=number, expression=text, reason="dynamic_variable")
                elif value.startswith("/") and re.search(r"root|prefix|aocl|fftw", name, re.IGNORECASE):
                    self.emit("prefix_declaration", path=value, variable=name, source=path,
                              line=number, status="static_candidate", conditional=conditional)
                    self.inspect_path("CMAKE_PREFIX_PATH", value, path, number)
                continue
            dependency = re.match(r"(?:prereq(?:-any)?|depends-on|conflict|module\s+load)\s+(.+)", text)
            lua_dependency = re.match(r"(?:prereq|prereq_any|depends_on|conflict|load)\s*\((.*)\)\s*$", text)
            if dependency or lua_dependency:
                relation = re.match(r"module\s+load|[A-Za-z_-]+", text).group(0)
                self.emit("compiler_requirement", source=path, line=number,
                          declaration=text, relation=relation,
                          status="declared_only_not_loaded", conditional=conditional)
                continue
            declaration = re.fullmatch(r"(?:setenv|prepend-path|append-path)\s+([A-Za-z_]\w*)\s+(.+)", text)
            if not declaration:
                declaration = re.fullmatch(r'(?:setenv|prepend_path|append_path)\s*\(\s*[\'"]([A-Za-z_]\w*)[\'"]\s*,\s*(.*)\)\s*', text)
            if declaration:
                name, expression = declaration.groups()
                if name not in PATH_VARIABLES:
                    continue
                self.emit("path_declaration", variable=name, source=path, line=number,
                          declaration=text, status="static_candidate", conditional=conditional)
                for value in self.path_values(expression, variables, path, number):
                    self.inspect_path(name, value, path, number)
                continue
            self.emit("unresolved", source=path, line=number, expression=text, reason="uninterpreted_module_statement")

    def run(self, module_roots, configs):
        self.emit("inventory_begin", mode="static_only_no_effective_environment")
        for root in self.env.get("MODULEPATH", "").split(":"):
            if root:
                self.add_root(root, "environment:MODULEPATH")
        for name in ("MODULESHOME", "LMOD_CMD"):
            value = self.env.get(name, "")
            if value:
                if value.startswith("/"):
                    self.module_home(name, value, "environment:" + name, 0)
                else:
                    self.emit("unresolved", variable=name, reason="nonabsolute_module_home")
        for root in module_roots:
            self.add_root(str(root), "argument:module-root")
        for path in CONFIGS:
            self.queue(path, 0, "standard_configuration_candidate")
        for name in self.names("/etc/profile.d"):
            if name.endswith(".sh") and ("module" in name.lower() or "lmod" in name.lower()):
                self.queue("/etc/profile.d/" + name, 0, "standard_configuration_directory")
        for path in configs:
            self.queue(str(path), 0, "argument:config")
        while self.pending:
            self.config(*self.pending.popleft())
        examined = set()
        # A declaration may expose one more module root; roots remain bounded.
        while len(examined) < len(self.roots):
            root = next(item for item in self.roots if item not in examined)
            examined.add(root)
            for module in MODULES:
                for suffix in ("", ".lua"):
                    self.module_file(str(Path(root) / (module + suffix)), module)
        counts = {}
        for record in self.records:
            counts[record["kind"]] = counts.get(record["kind"], 0) + 1
        found = sorted({record["module"] for record in self.records if record["kind"] == "modulefile"})
        missing = sorted(set(MODULES) - set(found))
        self.emit("inventory_summary", counts=counts, files_read=self.read_files,
                  bytes_read=self.read_bytes, effective_environment_verified=False,
                  compiler_link_verified=False, requested_modules=list(MODULES),
                  found_modules=found, missing_modules=missing,
                  unresolved=bool(missing or counts.get("unresolved") or counts.get("limit")))
        return self.records


def run_inventory(*, system_root=Path("/"), env=None, module_roots=(), configs=()):
    """Read static candidates; system_root is internal fixture injection only."""
    return Inventory(system_root, os.environ if env is None else env).run(module_roots, configs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-root", action="append", default=[])
    parser.add_argument("--config", action="append", default=[])
    args = parser.parse_args(argv)
    if os.environ.get("GPU_SUITE_FFTW_PROBE_APPROVED") != "yes":
        parser.error("static FFTW inventory requires separate owner approval")
    if any(not Path(path).is_absolute() for path in args.module_root + args.config):
        parser.error("module roots and config paths must be absolute observed paths")
    for record in run_inventory(module_roots=args.module_root, configs=args.config):
        print(json.dumps(record, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
