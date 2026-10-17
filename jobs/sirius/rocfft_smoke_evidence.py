#!/usr/bin/env python3
"""Validate already captured local build/ldd evidence; never execute a program."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Dict, Mapping, Optional, Sequence, Tuple

from gpu_suite.config import load_config
from gpu_suite.fftw_metadata import validate_fftw_threading
from gpu_suite.hashing import sha256_file
from gpu_suite.strict_json import dump_bytes, load


def resolved_libraries(text: str) -> Tuple[str, Dict[str, str]]:
    """Return address-independent ldd text and actual real-path content hashes."""
    if "not found" in text:
        raise ValueError("unresolved runtime library in ldd output")
    libraries = {}
    for line in text.splitlines():
        fields = line.split()
        resolved = (fields[2] if len(fields) > 2 and fields[1] == "=>"
                    else fields[0] if fields else "")
        if resolved.startswith("/"):
            library = Path(resolved).resolve(strict=True)
            if not library.is_file():
                raise ValueError("resolved dependency is not a regular file")
            libraries[str(library)] = sha256_file(library)
    # Preserve unnormalized output in the caller's command log, not identity.
    normalized = re.sub(r"\s+\(0x[0-9a-fA-F]+\)", "", text)
    return normalized, libraries


def _bounded_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("link evidence exceeds size limit: " + str(path))
    return content


def _linker_script_inputs(content: bytes) -> Sequence[str]:
    """Recognize a narrow GNU ld input grammar; never evaluate a script."""
    if len(content) > 65536:
        raise ValueError("linker script exceeds size limit")
    text = content.decode("ascii", errors="strict")
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"#[^\n]*", " ", text)
    tokens, offset = [], 0
    while offset < len(text):
        match = re.match(r"\s+|[A-Za-z0-9_./+\-]+|[(),;]", text[offset:])
        if match is None:
            raise ValueError("unsupported linker script syntax")
        token = match.group()
        if not token.isspace():
            tokens.append(token)
        offset += len(token)
    if len(tokens) > 1024:
        raise ValueError("linker script has too many tokens")
    references, index = [], 0

    def expect(value):
        nonlocal index
        if index >= len(tokens) or tokens[index] != value:
            raise ValueError("malformed linker script")
        index += 1

    def inputs(depth=0):
        nonlocal index
        if depth > 8:
            raise ValueError("linker script nesting limit exceeded")
        expect("(")
        count = 0
        while index < len(tokens) and tokens[index] != ")":
            token = tokens[index]
            index += 1
            if token == ",":
                continue
            if token == "AS_NEEDED":
                inputs(depth + 1)
            elif token.startswith("/") or re.fullmatch(r"-l[A-Za-z0-9_+.-]+", token):
                references.append(token)
            else:
                raise ValueError("unsupported linker input: " + token)
            count += 1
        expect(")")
        if not count:
            raise ValueError("empty linker input group")

    while index < len(tokens):
        directive = tokens[index]
        index += 1
        if directive in {"INPUT", "GROUP"}:
            inputs()
        elif directive == "OUTPUT_FORMAT":
            expect("(")
            formats = []
            while index < len(tokens) and tokens[index] != ")":
                token = tokens[index]
                index += 1
                if token != ",":
                    if not re.fullmatch(r"elf[A-Za-z0-9_-]+", token):
                        raise ValueError("unsupported linker output format")
                    formats.append(token)
            expect(")")
            if len(formats) not in (1, 3):
                raise ValueError("invalid linker output formats")
        else:
            raise ValueError("unsupported linker script directive: " + directive)
        if index < len(tokens) and tokens[index] == ";":
            index += 1
    if not references:
        raise ValueError("linker script has no library inputs")
    return references


def _link_dependency_paths(path: Path, script: Path):
    """Use the first CMake linker-depfile rule, not guessed library paths."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("link dependency file must be a regular non-symlink")
    content = _bounded_bytes(path, 1024 * 1024)
    text = content.decode("utf-8", errors="strict").replace("\\\n", "")
    rule = text.split("\n", 1)[0]
    if rule.count(":") != 1:
        raise ValueError("unsupported link dependency rule")
    target, dependencies = rule.split(":")
    # Current Sirius CMake emits simple paths without spaces or Make expansion.
    if not re.fullmatch(r"[A-Za-z0-9_./+\-]+", target):
        raise ValueError("unsupported link dependency target")
    paths = set()
    for token in dependencies.split():
        if not re.fullmatch(r"[A-Za-z0-9_./+\-]+", token):
            raise ValueError("unsupported link dependency path")
        candidate = Path(token)
        if candidate.is_absolute():
            paths.add(candidate.resolve(strict=True))
    if script not in paths:
        raise ValueError("link dependency file does not bind configured script")
    return paths, {"path": str(path.resolve(strict=True)),
                   "sha256": hashlib.sha256(content).hexdigest(), "target": target}


def _bind_linker_script(script: Path, digest: str, resolved: Mapping[str, str],
                        link_dependencies: Optional[Path]):
    content = _bounded_bytes(script, 65536)
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("configured linker script changed during verification")
    references = _linker_script_inputs(content)
    paths, depfile = None, None
    if any(reference.startswith("-l") for reference in references):
        if link_dependencies is None:
            raise ValueError("-l linker input requires recorded link dependency evidence")
        paths, depfile = _link_dependency_paths(Path(link_dependencies), script)
    inputs, dynamic_count = [], 0
    for reference in references:
        if reference.startswith("-l"):
            name = reference[2:]
            candidates = {path for path in paths if path.name == "lib" + name + ".a"
                          or re.fullmatch(r"lib" + re.escape(name) + r"\.so(?:\.[0-9]+)*", path.name)}
            if len(candidates) != 1 or next(iter(candidates)).suffix != ".a":
                raise ValueError("-l input is not one unambiguous recorded static archive: " + reference)
            library = next(iter(candidates))
        else:
            library = Path(reference).resolve(strict=True)
        if not library.is_file():
            raise ValueError("linker script input is not a regular file")
        with library.open("rb") as stream:
            magic = stream.read(8)
        actual_hash = sha256_file(library)
        if library.suffix == ".a" and magic == b"!<arch>\n":
            kind = "static-archive"
        elif magic.startswith(b"\x7fELF"):
            if resolved.get(str(library)) != actual_hash:
                raise ValueError("loaded linker-script dependency differs from referenced path/hash")
            dynamic_count += 1
            kind = "loaded-elf"
        else:
            raise ValueError("unsupported linker script input content")
        inputs.append({"reference": reference, "path": str(library), "sha256": actual_hash, "kind": kind})
    if not dynamic_count:
        raise ValueError("OpenMP linker script must bind a loaded ELF runtime")
    return {"script_path": str(script), "script_sha256": digest,
            "parsed_inputs": list(references), "inputs": inputs, "link_dependency_file": depfile}


def check_configured_dependencies(metadata: Mapping[str, object],
                                  resolved: Mapping[str, str],
                                  link_dependencies: Optional[Path] = None) -> Dict[str, object]:
    """Reject configured/loaded library drift; static archives bind the binary."""
    dependencies = metadata.get("dependencies")
    if not isinstance(dependencies, dict):
        raise ValueError("build metadata has no dependency evidence")
    if "fftw_threading" in metadata:
        validate_fftw_threading(metadata)
    profile = metadata.get("build_profile")
    if profile in {"hip", "openmp"} and not {"hip", "rocfft"}.issubset(dependencies):
        raise ValueError("GPU build metadata lacks HIP/rocFFT dependency evidence")
    linker_inputs = {}
    for name, dependency in dependencies.items():
        if not isinstance(dependency, dict):
            raise ValueError("malformed dependency evidence: " + name)
        configured = Path(dependency["path"]).resolve(strict=True)
        actual_hash = sha256_file(configured)
        if actual_hash != dependency.get("sha256"):
            raise ValueError("configured dependency changed since build: " + name)
        # An archive is linked into the immutable executable, not a loaded DSO.
        if configured.suffix == ".a":
            continue
        if resolved.get(str(configured)) != actual_hash:
            if re.fullmatch(r"fftw_openmp_runtime_[0-9]+", name):
                with configured.open("rb") as stream:
                    prefix = stream.read(4)
                if prefix != b"\x7fELF":
                    try:
                        linker_inputs[name] = _bind_linker_script(configured, actual_hash, resolved,
                                                                 link_dependencies)
                    except (OSError, ValueError) as error:
                        raise ValueError("loaded dependency differs from configured path/hash: " + name
                                         + "; FFTW OpenMP linker input " + str(configured)
                                         + ": " + str(error)) from error
                    continue
            raise ValueError("loaded dependency differs from configured path/hash: " + name)
    return linker_inputs


def prepare(root: Path, config_path: Path, hip_device: int, openmp_device: int) -> str:
    config = load_config(config_path)
    evidence = {
        "kind": "rocfft-allocation-smoke",
        "hip_device": hip_device, "openmp_device": openmp_device,
        "environment": {name: os.environ.get(name) for name in (
            "ROCM_PATH", "HIP_PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH",
            "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL", "OMP_DEFAULT_DEVICE",
            "OMP_TARGET_OFFLOAD", "OMP_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND",
            "OMP_DYNAMIC", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS", "MKL_DYNAMIC",
            "MKL_THREADING_LAYER", "OPENBLAS_NUM_THREADS")},
        "build_metadata": {}, "resolution": {}, "resolved_library_sha256": {},
        "linker_input_bindings": {},
    }
    for profile in ("cpu", "hip", "openmp"):
        metadata = load(root / profile / "build-metadata.json")
        paths = list((root / "logs").glob("*-resolved-" + profile + ".stdout"))
        if len(paths) != 1:
            raise ValueError("expected exactly one resolution log for " + profile)
        normalized, libraries = resolved_libraries(paths[0].read_text())
        evidence["linker_input_bindings"][profile] = check_configured_dependencies(metadata, libraries)
        evidence["build_metadata"][profile] = metadata
        evidence["resolution"][profile] = normalized
        evidence["resolved_library_sha256"][profile] = libraries
    # Exclusive outputs only after the complete evidence passed all checks.
    with (root / "config.json").open("xb") as stream:
        stream.write(config_path.read_bytes())
    runtime = dump_bytes(evidence)
    with (root / "runtime-environment.json").open("xb") as stream:
        stream.write(runtime)
    for fmt in ("jsonl", "csv"):
        with (root / ("config-" + fmt + ".json")).open("xb") as stream:
            stream.write(dump_bytes(dict(config, output_format=fmt)))
    return hashlib.sha256(runtime).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("hip_device", type=int)
    parser.add_argument("openmp_device", type=int)
    args = parser.parse_args(argv)
    try:
        if args.hip_device < 0 or args.openmp_device < 0:
            raise ValueError("device indices must be nonnegative")
        print(prepare(args.root, args.config, args.hip_device, args.openmp_device))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("rocFFT smoke evidence rejected: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
