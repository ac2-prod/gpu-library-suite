#!/usr/bin/env python3
"""Bounded, read-only FFTW threading evidence from one explicit AOCL prefix.

This never loads a module/library or compiles, links, or executes a test. Symbol
evidence is a candidate for later approved link/runtime checks, not support proof.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import stat
import subprocess
import sys
import time


MAX_DIRECTORY_ENTRIES = 512
MAX_LIBRARIES = 32
MAX_LIBRARY_BYTES = 64 * 1024 * 1024
MAX_TOTAL_LIBRARY_BYTES = 256 * 1024 * 1024
MAX_TEXT_BYTES = 128 * 1024
MAX_COMMAND_BYTES = 4 * 1024 * 1024
COMMAND_TIMEOUT = 10
MAX_COLLECTION_SECONDS = 90
LIBRARY_DIRS = ("lib", "lib64", "lib_LP64")
PACKAGE_DIRS = ("lib/pkgconfig", "lib64/pkgconfig", "lib_LP64/pkgconfig", "share/pkgconfig")
HEADER_FILES = ("include/fftw3.h", "include_LP64/fftw3.h")
LIBRARY_NAME = re.compile(r"libfftw3f(?P<variant>_threads|_omp)?\.(?:so(?:\.[0-9.]+)?|a|dylib)\Z")
PACKAGE_NAME = re.compile(r"fftw3f(?:_threads|_omp)?\.pc\Z")
THREAD_APIS = ("fftwf_init_threads", "fftwf_plan_with_nthreads", "fftwf_cleanup_threads")
BASE_APIS = ("fftwf_plan_many_dft", "fftwf_execute", "fftwf_destroy_plan")
APIS = THREAD_APIS + BASE_APIS


def bounded_command(argv):
    """Read at most MAX_COMMAND_BYTES from a bounded-duration utility child."""
    record = {"argv": list(argv), "status": "UNKNOWN", "exit": None,
              "stdout": "", "stderr": ""}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    environment = dict(os.environ, LC_ALL="C")
    try:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env=environment, shell=False)
    except OSError as error:
        record.update(status="UNAVAILABLE", reason=type(error).__name__)
        return record
    deadline = time.monotonic() + COMMAND_TIMEOUT
    reason = None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ, "stdout")
            selector.register(child.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fd, 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    available = MAX_COMMAND_BYTES - sum(map(len, buffers.values()))
                    buffers[key.data].extend(data[:available])
                    if len(data) > available:
                        reason = "output_limit"
                        break
                if reason:
                    break
            if reason:
                child.kill()
            try:
                record["exit"] = child.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                reason = "timeout"
                child.kill()
                record["exit"] = child.wait(timeout=1)
    except (OSError, subprocess.TimeoutExpired) as error:
        reason = type(error).__name__
        child.kill()
        try:
            record["exit"] = child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    finally:
        child.stdout.close()
        child.stderr.close()
    for name, value in buffers.items():
        record[name] = bytes(value).decode("utf-8", errors="replace")
    record["status"] = "COMPLETE" if reason is None and record["exit"] == 0 else "INCOMPLETE"
    if reason:
        record["reason"] = reason
    return record


def readelf_symbols(text):
    """Only externally visible, defined function symbols count as exports."""
    result = {name: {"defined_export": False, "undefined_reference": False} for name in APIS}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].rstrip(":").isdigit():
            continue
        name = fields[7].split("@", 1)[0]
        if name not in result:
            continue
        if fields[6] == "UND":
            result[name]["undefined_reference"] = True
        elif (fields[3] in ("FUNC", "IFUNC", "GNU_IFUNC") and
              fields[4] in ("GLOBAL", "WEAK") and fields[5] in ("DEFAULT", "PROTECTED")):
            result[name]["defined_export"] = True
    return result


def nm_symbols(text):
    result = {name: {"defined_export": False, "undefined_reference": False} for name in APIS}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("@", 1)[0]
        if name not in result:
            continue
        if fields[1] in ("U", "w", "v"):
            result[name]["undefined_reference"] = True
        elif fields[1] in ("T", "W", "i"):
            result[name]["defined_export"] = True
    return result


def compact_command(record):
    """Retain relevant evidence and a digest, not thousands of FFT codelets."""
    result = dict(record)
    data = record["stdout"].encode("utf-8")
    result["stdout_bytes"] = len(data)
    result["stdout_sha256"] = hashlib.sha256(data).hexdigest()
    result["stdout_filtered"] = True
    result["stdout"] = "\n".join(line for line in record["stdout"].splitlines()
                                    if "(NEEDED)" in line or any(name in line for name in APIS))
    return result


class Inventory:
    def __init__(self, root, runner=bounded_command, locate=shutil.which):
        supplied = Path(root)
        if not supplied.is_absolute() or ".." in supplied.parts:
            raise ValueError("--aocl-root must be an explicit absolute directory without '..'")
        self.root = supplied.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("--aocl-root must be an existing directory")
        self.runner = runner
        self.tools = {name: locate(name) for name in ("readelf", "nm")}
        self.issues = []
        self.bytes_hashed = 0
        self.inspected = {}
        self.deadline = time.monotonic() + MAX_COLLECTION_SECONDS

    def query(self, argv):
        if self.deadline - time.monotonic() < COMMAND_TIMEOUT:
            return dict(argv=list(argv), status="INCOMPLETE", exit=None,
                        stdout="", stderr="", reason="collection_time_limit")
        return self.runner(argv)

    def issue(self, path, reason):
        self.issues.append({"path": str(path), "reason": reason})

    def contained(self, path):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.root)
            return resolved
        except (OSError, ValueError, RuntimeError) as error:
            self.issue(path, "missing_or_outside_root:" + type(error).__name__)
            return None

    def names(self, relative):
        path = self.root / relative
        if not path.exists():
            return []
        resolved = self.contained(path)
        if resolved is None:
            return []
        names = []
        try:
            with os.scandir(resolved) as entries:
                for entry in entries:
                    if len(names) >= MAX_DIRECTORY_ENTRIES:
                        self.issue(path, "directory_entry_limit")
                        return []  # Never represent a partial directory as complete.
                    names.append(entry.name)
        except OSError as error:
            self.issue(path, "unreadable_directory:" + type(error).__name__)
            return []
        return sorted(names)

    def text_file(self, relative):
        path = self.root / relative
        record = {"path": str(path), "status": "MISSING"}
        if not path.exists():
            return record
        resolved = self.contained(path)
        if resolved is None:
            record["status"] = "UNKNOWN"
            return record
        record["real_path"] = str(resolved)
        try:
            metadata = resolved.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_TEXT_BYTES:
                raise ValueError("not_regular_or_text_limit")
            with resolved.open("rb") as stream:
                data = stream.read(MAX_TEXT_BYTES + 1)
            if len(data) > MAX_TEXT_BYTES:
                raise ValueError("text_grew_beyond_limit")
            record.update(status="PRESENT", sha256=hashlib.sha256(data).hexdigest(),
                          text=data.decode("utf-8"))
        except (OSError, ValueError) as error:
            record.update(status="UNKNOWN", reason=str(error))
            self.issue(path, "unreadable_or_oversized_text")
        return record

    def symbols(self, path):
        result = {"status": "UNKNOWN", "needed_status": "UNKNOWN", "needed": [], "commands": []}
        if self.tools["readelf"]:
            record = self.query([self.tools["readelf"], "--dyn-syms", "--wide", str(path)])
            result["commands"].append(compact_command(record))
            if record["status"] == "COMPLETE":
                result.update(status="INSPECTED", source="readelf", symbols=readelf_symbols(record["stdout"]))
            dynamic = self.query([self.tools["readelf"], "--dynamic", "--wide", str(path)])
            result["commands"].append(compact_command(dynamic))
            if dynamic["status"] == "COMPLETE":
                result["needed"] = re.findall(r"\(NEEDED\).*?Shared library: \[([^\]]+)\]", dynamic["stdout"])
                result["needed_status"] = "INSPECTED"
        if result["status"] == "UNKNOWN" and self.tools["nm"]:
            record = self.query([self.tools["nm"], "-D", "--format=posix", str(path)])
            result["commands"].append(compact_command(record))
            if record["status"] == "COMPLETE":
                result.update(status="INSPECTED", source="nm", symbols=nm_symbols(record["stdout"]))
        if result["status"] == "INSPECTED":
            result["all_thread_api_exports"] = all(result["symbols"][name]["defined_export"] for name in THREAD_APIS)
            result["all_base_api_exports"] = all(result["symbols"][name]["defined_export"] for name in BASE_APIS)
        else:
            result["reason"] = "no_successful_symbol_inspection"
        return result

    def library(self, path, variant):
        record = {"path": str(path), "variant": variant, "status": "UNKNOWN"}
        resolved = self.contained(path)
        if resolved is None:
            return record
        record["real_path"] = str(resolved)
        if resolved in self.inspected:
            record.update(self.inspected[resolved])
            return record
        try:
            if time.monotonic() >= self.deadline:
                raise ValueError("collection_time_limit")
            before = resolved.stat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("not_regular")
            if before.st_size > MAX_LIBRARY_BYTES or self.bytes_hashed + before.st_size > MAX_TOTAL_LIBRARY_BYTES:
                raise ValueError("library_byte_limit")
            digest = hashlib.sha256()
            count = 0
            with resolved.open("rb") as stream:
                magic = stream.read(4)
                digest.update(magic)
                count = len(magic)
                self.bytes_hashed += count
                while True:
                    if time.monotonic() >= self.deadline:
                        raise ValueError("collection_time_limit")
                    data = stream.read(min(1024 * 1024, MAX_LIBRARY_BYTES - count + 1))
                    if not data:
                        break
                    count += len(data)
                    self.bytes_hashed += len(data)
                    if count > MAX_LIBRARY_BYTES or self.bytes_hashed > MAX_TOTAL_LIBRARY_BYTES:
                        raise ValueError("library_grew_beyond_limit")
                    digest.update(data)
            after = resolved.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("library_changed_during_read")
            record.update(status="PRESENT", sha256=digest.hexdigest(), bytes=count,
                          format="ELF" if magic == b"\x7fELF" else "NON_ELF")
            if magic == b"\x7fELF":
                record["inspection"] = self.symbols(resolved)
                final = resolved.stat()
                if (after.st_ino, after.st_size, after.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns):
                    raise ValueError("library_changed_during_inspection")
            else:
                record["inspection"] = {"status": "UNKNOWN", "reason": "non_elf_static_or_other_format"}
            self.inspected[resolved] = {key: value for key, value in record.items()
                                        if key not in ("path", "variant", "real_path")}
        except (OSError, ValueError) as error:
            record.update(status="UNKNOWN", reason=str(error))
            self.issue(path, str(error))
        return record

    def collect(self):
        headers = [self.text_file(relative) for relative in HEADER_FILES]
        for header in headers:
            if "text" in header:
                # FFTW's public header uses FFTW_DEFINE_API / X(name) macros.
                text = header.pop("text")
                header["thread_api_tokens"] = {
                    name: bool(re.search(r"\b" + name.removeprefix("fftwf_") + r"\b", text))
                    for name in THREAD_APIS}
                header["declaration_status"] = "TEXT_ONLY_NOT_PREPROCESSED"
        packages = []
        for directory in PACKAGE_DIRS:
            for name in self.names(directory):
                if PACKAGE_NAME.fullmatch(name):
                    record = self.text_file(str(Path(directory) / name))
                    if record["status"] == "PRESENT":
                        record["version_text"] = next((line.partition(":")[2].strip()
                            for line in record["text"].splitlines() if line.startswith("Version:")), None)
                        record["status"] = "STATIC_TEXT_NOT_EVALUATED"
                    packages.append(record)
        libraries = []
        for directory in LIBRARY_DIRS:
            for name in self.names(directory):
                match = LIBRARY_NAME.fullmatch(name)
                if match:
                    if len(libraries) >= MAX_LIBRARIES:
                        self.issue(self.root / directory, "library_count_limit")
                        break
                    libraries.append(self.library(self.root / directory / name,
                        {None: "base", "_threads": "pthread", "_omp": "openmp"}[match.group("variant")]))
        incomplete = bool(self.issues) or any(
            entry["status"] != "PRESENT" or entry["inspection"]["status"] != "INSPECTED" or
            entry["inspection"].get("needed_status") != "INSPECTED" for entry in libraries)
        return {"schema_version": 1, "aocl_root": str(self.root),
                "status": "INCOMPLETE" if incomplete else "COMPLETE_STATIC_INVENTORY",
                "scope": "bounded prefix files; no module evaluation, compilation, linking, library loading or execution",
                "limitations": ["Symbol exports do not prove compile/link/runtime usability.",
                    "A shared-library filename is not an upstream FFTW release version.",
                    "Utility stdout is filtered to the selected symbols and NEEDED entries after bounded inspection.",
                    "Missing files in these bounded directories do not prove the site has no threaded FFTW."],
                "tools": self.tools, "directories": list(LIBRARY_DIRS + PACKAGE_DIRS),
                "limits": {"collection_seconds": MAX_COLLECTION_SECONDS,
                           "command_seconds": COMMAND_TIMEOUT, "command_bytes": MAX_COMMAND_BYTES,
                           "libraries": MAX_LIBRARIES, "library_bytes": MAX_LIBRARY_BYTES,
                           "total_library_bytes": MAX_TOTAL_LIBRARY_BYTES},
                "headers": headers, "pkg_config": packages, "libraries": libraries,
                "issues": self.issues}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aocl-root", required=True, help="explicit existing absolute AOCL provider directory")
    arguments = parser.parse_args(argv)
    try:
        report = Inventory(arguments.aocl_root).collect()
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "STOPPED", "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["status"] == "COMPLETE_STATIC_INVENTORY" else 1


if __name__ == "__main__":
    sys.exit(main())
