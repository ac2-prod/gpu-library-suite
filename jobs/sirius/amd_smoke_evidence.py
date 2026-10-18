"""Read-only six-family diagnostic evidence; never launch or load a program.

ldd describes loader resolution. LD_DEBUG=libs describes initialization events
in the traced child, not symbol ownership, all mappings, or CPU worker counts.
The latter is diagnostic smoke evidence, never performance-measurement input.
"""

import hashlib
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Mapping, Optional, Union

from gpu_suite.fftw_metadata import validate_fftw_threading
from rocfft_smoke_evidence import _linker_script_inputs


FAMILIES = {"rocfft", "rocblas", "rocsparse", "rocsolver", "rocrand", "rocthrust"}
HEADER_NAMES = {"aocl_sparse_header", "rocthrust", "rocprim"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _absolute(value):
    if (not isinstance(value, str) or not value.startswith("/")
            or any(ord(c) < 32 or ord(c) == 127 for c in value) or " (deleted)" in value):
        raise ValueError("unsafe or nonabsolute evidence path")
    return Path(value)


def _file_identity(path):
    """Hash one resolved regular file, rejecting changes during this read."""
    resolved = _absolute(str(path)).resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("evidence is not a regular file")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("evidence changed before opening")
        first = stream.read(8)
        digest.update(first)
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        finished = os.fstat(stream.fileno())
    after = resolved.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns)
    if signature(before) != signature(finished) or signature(before) != signature(after):
        raise ValueError("evidence changed while hashing")
    return {"path": str(resolved), "sha256": digest.hexdigest()}, first


def _kind(name, path, magic):
    if name in HEADER_NAMES:
        if magic.startswith(b"\x7fELF") or magic == b"!<arch>\n":
            raise ValueError("header identity is not a header: " + name)
        return "header"
    if magic.startswith(b"\x7fELF"):
        return "elf"
    if magic == b"!<arch>\n":
        return "static-archive"
    with Path(path).open("rb") as stream:
        content = stream.read(65537)
    _linker_script_inputs(content)
    return "linker-script"


def _provider_names(dependencies, inputs, prefix, interface):
    links = inputs.get(interface + "_INTERFACE_LINK_LIBRARIES")
    if not isinstance(links, str) or not links:
        raise ValueError("missing ordered provider link interface")
    paths = links.split(";")
    if not Path(paths[0]).is_absolute():
        raise ValueError("primary provider must be an absolute link input")
    selected = []
    for index, path in enumerate(paths):
        if not Path(path).is_absolute():
            if path not in {"", "Threads::Threads", "dl", "m", "pthread"}:
                raise ValueError("unresolved provider link input")
            continue
        name = prefix if index == 0 else prefix + "_link_" + str(index)
        identity = dependencies.get(name)
        if not isinstance(identity, Mapping) or identity.get("path") != str(_absolute(path).resolve(strict=True)):
            raise ValueError("ordered provider dependency differs: " + name)
        selected.append(name)
    actual = {name for name in dependencies if name == prefix or name.startswith(prefix + "_link_")}
    if actual != set(selected):
        raise ValueError("provider link dependency inventory differs")
    return selected


def select_artifact_dependencies(entry: Mapping, metadata: Mapping):
    """Select this artifact's configured inputs; unrelated profile inputs stay out.

    Both roles are accepted. A serial FFTW teaching binary does not link the
    thread provider; a dual-backend benchmark does, even on a serial invocation.
    Callers must also validate the complete manifest and its metadata hash.
    """
    family, profile = entry.get("library"), entry.get("implementation")
    if (family not in FAMILIES or profile not in {"cpu", "hip", "openmp"}
            or entry.get("build_profile") != profile
            or entry.get("executable_role") not in {"example", "benchmark"}
            or metadata.get("vendor") != "amd" or metadata.get("build_profile") != profile):
        raise ValueError("invalid AMD artifact family/profile/role")
    dependencies, inputs = metadata.get("dependencies"), metadata.get("amd_build_inputs", {})
    if not isinstance(dependencies, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("missing dependency metadata")
    backends, variant = entry.get("supported_cpu_backends"), entry.get("backend_variant")
    if (not isinstance(backends, list) or any(not isinstance(b, str) for b in backends)
            or len(set(backends)) != len(backends)):
        raise ValueError("invalid supported CPU backends")
    names = []
    if profile != "cpu":
        if backends or variant != profile + "-" + family:
            raise ValueError("GPU artifact/backend mismatch")
        names = ["hip", family]
        if family == "rocsolver":
            names.append("rocblas")
        if family == "rocthrust":
            names.append("rocprim")
    elif family == "rocfft":
        threaded = "cpu-fftw-threaded" in backends
        expected = ["cpu-fftw-serial", "cpu-fftw-threaded"] if threaded else ["cpu-fftw-serial"]
        if (backends != expected or variant != ("fftw3f-serial+threaded" if threaded else "fftw3f-serial")
                or (threaded and entry["executable_role"] != "benchmark")):
            raise ValueError("FFTW artifact/backend mismatch")
        validate_fftw_threading(metadata, require_threaded=threaded)
        names = ([name for name in dependencies if name.startswith("fftw")]
                 if threaded else ["fftw3f"])
    elif family in {"rocblas", "rocsolver"}:
        base, interface, option = (("cblas", "CBLAS", "GPU_SUITE_CPU_BLAS_BACKEND")
                                   if family == "rocblas" else
                                   ("lapacke", "LAPACKE", "GPU_SUITE_CPU_LAPACK_BACKEND"))
        providers = {"generic_" + base: "cpu-generic-" + base,
                     "openblas": "cpu-openblas", "onemkl": "cpu-onemkl"}
        if variant not in providers or backends != [providers[variant]] or inputs.get(option) != variant.upper():
            raise ValueError("CPU provider selection differs from artifact")
        names = _provider_names(dependencies, inputs, base, interface)
    elif family == "rocsparse":
        choices = {"aocl-sparse": ("AOCLSPARSE", "cpu-aocl-sparse", "aocl_sparse", "AOCLSparse"),
                   "onemkl-sparse": ("ONEMKL", "cpu-onemkl", "mkl_sparse", "MKLSparse"),
                   "reference-csr": ("REFERENCE", "cpu-reference-csr", None, None)}
        if variant not in choices:
            raise ValueError("unknown Sparse artifact backend")
        choice, backend, prefix, interface = choices[variant]
        if backends != [backend] or inputs.get("GPU_SUITE_CPU_SPARSE_BACKEND") != choice:
            raise ValueError("Sparse provider selection differs from artifact")
        if prefix:
            names = _provider_names(dependencies, inputs, prefix, interface)
        if prefix == "aocl_sparse":
            names.append("aocl_sparse_header")
            include = inputs.get("AOCLSparse_INTERFACE_INCLUDE_DIRECTORIES")
            header = dependencies.get("aocl_sparse_header", {})
            if (not isinstance(include, str) or not isinstance(header, Mapping)
                    or str((_absolute(include) / "aoclsparse.h").resolve(strict=True)) != header.get("path")):
                raise ValueError("AOCL Sparse consumed header identity differs")
    elif family == "rocrand":
        if variant != "std-random-serial" or backends != ["cpu-std-random-serial"]:
            raise ValueError("random CPU artifact/backend mismatch")
    else:
        if variant == "stl-serial" and backends == ["cpu-stl-serial"]:
            if inputs.get("GPU_SUITE_CPU_THRUST_BACKEND") != "STL":
                raise ValueError("STL selection mismatch")
        elif variant == "openmp" and backends == ["cpu-openmp"]:
            if inputs.get("GPU_SUITE_CPU_THRUST_BACKEND") != "OPENMP":
                raise ValueError("OpenMP CPU selection mismatch")
            names = sorted(name for name in dependencies if name.startswith("cpu_openmp_runtime_"))
            if not names or set(names) != {"cpu_openmp_runtime_" + str(i) for i in range(len(names))}:
                raise ValueError("missing/invalid CPU OpenMP runtime dependencies")
        else:
            raise ValueError("reduction CPU artifact/backend mismatch")
    selected = {}
    for name in names:
        expected = dependencies.get(name)
        if (not isinstance(expected, Mapping) or set(expected) != {"path", "sha256"}
                or not isinstance(expected["sha256"], str) or SHA256.fullmatch(expected["sha256"]) is None):
            raise ValueError("missing/invalid configured identity: " + name)
        actual, magic = _file_identity(expected["path"])
        if actual != expected:
            raise ValueError("configured identity changed or is not canonical: " + name)
        selected[name] = dict(actual, kind=_kind(name, actual["path"], magic))
    return selected


def _recorded_bytes(identity, maximum):
    """Read the exact hash-approved regular log, rejecting symlink aliases."""
    if (not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}
            or not isinstance(identity["sha256"], str) or SHA256.fullmatch(identity["sha256"]) is None):
        raise ValueError("invalid recorded link evidence identity")
    path = _absolute(identity["path"])
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("recorded link evidence must be canonical and non-symlink")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise ValueError("recorded link evidence is not a bounded regular file")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        content = stream.read(maximum + 1)
        finished = os.fstat(stream.fileno())
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns)
    if (signature(before) != signature(opened) or signature(before) != signature(finished)
            or signature(before) != signature(path.stat()) or len(content) != before.st_size
            or hashlib.sha256(content).hexdigest() != identity["sha256"]):
        raise ValueError("recorded link evidence hash/identity changed")
    return content.decode("utf-8", errors="strict")


def _link_argv(line):
    # This is a parser, never an evaluator. Current Sirius commands use these
    # plain tokens; response files, expansion and shell syntax need new review.
    if any(character in line for character in "\x00\r\n;$`&|<>"):
        raise ValueError("shell syntax in recorded link command")
    tokens = shlex.split(line, comments=False, posix=True)
    if not tokens or any(re.fullmatch(r"[A-Za-z0-9_./:+,=%-]+", token) is None for token in tokens):
        raise ValueError("response file or ambiguous recorded link token")
    return tokens


def _hip_link_comparison_argv(argv, program):
    """Ignore only this target's optional dependency-output pair for equality.

    Keep the original argv in the evidence. This does not read link.d or claim
    that a depfile was produced; all code-generating/link-input options remain.
    """
    result, index, seen = [], 0, False
    expected = "--dependency-file=CMakeFiles/" + program.name + ".dir/link.d"
    while index < len(argv):
        if argv[index] == "-Xlinker":
            if seen or argv[index:index + 2] != ["-Xlinker", expected]:
                raise ValueError("unreviewed or duplicate HIP linker dependency-output option")
            seen = True
            index += 2
        else:
            result.append(argv[index])
            index += 1
    return result


def _hip_link_inputs(argv, compiler, program):
    """Validate current recorded HIP argv without opening a library or object."""
    if argv[0] != compiler or argv.count("-o") != 1 or "--hip-link" not in argv or "-c" in argv:
        raise ValueError("recorded link compiler/output kind differs")
    output_index = argv.index("-o") + 1
    if output_index == len(argv) or argv[output_index] not in {program.name, str(program)}:
        raise ValueError("recorded link output differs from artifact")
    _hip_link_comparison_argv(argv, program)  # Validate any -Xlinker pair first.
    inputs, index = [], 1
    while index < len(argv):
        token = argv[index]
        if token in {"-o", "-Xlinker"}:
            index += 2
            continue
        if token.startswith("-"):
            if re.fullmatch(r"(?:-O[0123sg]|-D[A-Za-z0-9_=]+|--hip-link|"
                            r"--offload-arch=[A-Za-z0-9_:+,-]+|--rtlib=[A-Za-z0-9_-]+|"
                            r"-unwindlib=[A-Za-z0-9_-]+|-l[A-Za-z0-9_.+-]+|-L/[^\s]+)", token) is None:
                raise ValueError("unreviewed recorded HIP link option")
            # -l and -L remain raw evidence only; never guess their resolution.
        else:
            if re.search(r"(?:\.o|\.a|\.so(?:\.[0-9]+)*)$", token) is None:
                raise ValueError("ambiguous recorded HIP link input")
            inputs.append((index, token))
        index += 1
    if not inputs:
        raise ValueError("recorded HIP link command lacks explicit inputs")
    return inputs


def _native_hip_link_evidence(entry, metadata, selected, descriptor):
    if (set(descriptor) != {"mode", "link_command", "build_stdout", "build_directory"}
            or descriptor["mode"] != "native-hip-link-command"
            or entry.get("implementation") != "hip" or entry.get("compiler_language") != "hip"):
        raise ValueError("recorded native-HIP link mode requires a HIP artifact")
    program = _absolute(entry["executable_path"])
    directory = _absolute(descriptor["build_directory"])
    if (directory != program.parent or directory.resolve(strict=True) != directory
            or not directory.is_dir() or program.name != entry.get("target_name")):
        raise ValueError("recorded link working directory/target differs")
    command_path = directory / "CMakeFiles" / (entry["target_name"] + ".dir") / "link.txt"
    text = _recorded_bytes(descriptor["link_command"], 1024 * 1024)
    if descriptor["link_command"].get("path") != str(command_path):
        raise ValueError("recorded link command belongs to another artifact")
    compiler = metadata.get("hip", {}).get("compiler_path")
    _absolute(compiler)
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("expected exactly one recorded HIP link command")
    argv = _link_argv(lines[0])
    input_tokens = _hip_link_inputs(argv, compiler, program)
    stdout = _recorded_bytes(descriptor["build_stdout"], 16 * 1024 * 1024)
    matches, completions = [], []
    for index, line in enumerate(stdout.splitlines()):
        if re.fullmatch(r"\[100%\]\s+Built target " + re.escape(program.name), line.strip()):
            completions.append(index)
        try:
            candidate = shlex.split(line, comments=False, posix=True)
        except ValueError:
            continue  # Unrelated human-readable build output is not a command.
        if not candidate or candidate[0] != compiler or "-c" in candidate:
            continue
        matches.append((index, _link_argv(line)))
    comparison_argv = _hip_link_comparison_argv(argv, program)
    if (len(matches) != 1
            or _hip_link_comparison_argv(matches[0][1], program) != comparison_argv
            or len(completions) != 1
            or completions[0] <= matches[0][0]):
        raise ValueError("successful build stdout lacks one identical HIP link invocation")
    _hip_link_inputs(matches[0][1], compiler, program)
    inputs, paths = [], set()
    for index, token in input_tokens:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = directory / candidate
        identity, magic = _file_identity(candidate)
        if not (magic.startswith(b"\x7fELF") or magic == b"!<arch>\n"):
            # Linker scripts require the depfile parser's explicit input proof;
            # this command mode must not invent their transitive dependencies.
            raise ValueError("native HIP explicit input is not an ELF object or archive")
        inputs.append(dict(identity, token=token, argv_index=index))
        paths.add(Path(identity["path"]))
    if not inputs or any(Path(identity["path"]) not in paths for identity in selected.values()
                         if identity["kind"] != "header"):
        raise ValueError("configured dependency absent from recorded HIP link invocation")
    return paths, {"mode": "native-hip-link-command", "target": str(program),
                   "link_command": dict(descriptor["link_command"]),
                   "build_stdout": dict(descriptor["build_stdout"]),
                   "build_directory": str(directory), "argv": argv,
                   "build_argv": matches[0][1], "comparison_argv": comparison_argv,
                   "comparison_rule": "ignore only optional target-local -Xlinker --dependency-file pair",
                   "ordered_inputs": inputs,
                   "scope": "explicit inputs of recorded successful link command; not linker depfile or inferred -l resolution"}


def _parse_link_dependency_rule(text, program):
    """Parse Sirius' simple Make depfiles without resolving any prerequisite.

    The first rule owns the link inputs. Later empty rules are only Make's
    phony targets for those inputs, not another source of prerequisites. Keep
    ordering, aliases and repetitions exactly as recorded by the linker.
    """
    if (not isinstance(text, str) or len(text.encode("utf-8")) > 1024 * 1024
            or any(ord(c) < 32 and c not in "\n\t" or ord(c) == 127 for c in text)):
        raise ValueError("invalid or oversized link dependency file")
    lines = text.replace("\\\n", " ").splitlines()
    if not lines or lines[0].count(":") != 1:
        raise ValueError("unsupported link dependency rule")
    target, dependencies = lines[0].split(":")
    token_pattern = r"[A-Za-z0-9_./+\-]+"
    if re.fullmatch(token_pattern, target) is None:
        raise ValueError("unsupported link dependency target")
    if target not in {program.name, str(program)}:
        raise ValueError("link dependency file belongs to another artifact")
    tokens = dependencies.split()
    if not tokens or any(re.fullmatch(token_pattern, token) is None for token in tokens):
        raise ValueError("unsupported or empty link dependency path list")
    token_set = set(tokens)
    for line in lines[1:]:
        if not line.strip():
            continue
        phony = re.fullmatch("(" + token_pattern + r"):[ \t]*", line)
        if phony is None or phony[1] not in token_set:
            raise ValueError("unsupported trailing link dependency rule")
    return target, tokens


def _artifact_link_dependencies(path, program):
    """Separate historical object names from current persistent file proof.

    Compiler-created OpenMP image-wrapper objects normally disappear after
    linking. An .o token is historical input evidence only, even if its file
    still exists. It is never opened, hashed or used to satisfy a configured
    library or linker-script input. Every other prerequisite must still exist
    as a regular file. Relative paths use the artifact's recorded link cwd.
    """
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("link dependency file must be a bounded regular non-symlink")
    identity, _ = _file_identity(path)
    text = _recorded_bytes(identity, 1024 * 1024)
    target, tokens = _parse_link_dependency_rule(text, program)
    paths, prerequisites = set(), []
    for token in tokens:
        if token.endswith(".o"):
            prerequisites.append({"token": token, "kind": "historical-object",
                                  "current_file_verified": False})
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = program.parent / candidate
        actual, _ = _file_identity(candidate)
        paths.add(Path(actual["path"]))
        prerequisites.append(dict(actual, token=token, kind="persistent-input",
                                  current_file_verified=True))
    return paths, dict(identity, target=target, build_directory=str(program.parent),
                       raw_prerequisites=tokens, prerequisites=prerequisites,
                       scope="recorded linker prerequisites; object tokens are historical, not current file identities")


def _link_evidence(entry, metadata, selected, link_dependencies):
    if isinstance(link_dependencies, Mapping):
        return _native_hip_link_evidence(entry, metadata, selected, link_dependencies)
    linked = [identity for identity in selected.values() if identity["kind"] != "header"]
    if not linked and link_dependencies is None:
        return set(), None
    if link_dependencies is None:
        raise ValueError("actual artifact link dependency file is required")
    # CMake's relative first target is the executable filename; absolute targets
    # must match the full immutable artifact path.
    program = _absolute(entry["executable_path"])
    paths, evidence = _artifact_link_dependencies(Path(link_dependencies), program)
    if any(Path(identity["path"]) not in paths for identity in linked):
        raise ValueError("configured input absent from actual link dependency file")
    return paths, evidence


def validate_link_evidence(entry: Mapping, metadata: Mapping, descriptor):
    """Preflight explicit link proof before a child starts; no process is run.

    The caller obtains HIP descriptors only from the verified successful build
    proof. A missing depfile never selects this mode automatically. CPU/OpenMP
    retain their existing depfile contract, including linker-script inputs.
    """
    selected = select_artifact_dependencies(entry, metadata)
    paths, depfile = _link_evidence(entry, metadata, selected, descriptor)
    bindings = {}
    for name, identity in selected.items():
        kind, path = identity["kind"], identity["path"]
        binding = dict(identity)
        if kind == "linker-script":
            with Path(path).open("rb") as stream:
                content = stream.read(65537)
            if hashlib.sha256(content).hexdigest() != identity["sha256"]:
                raise ValueError("linker script changed during verification")
            inputs = []
            for reference in _linker_script_inputs(content):
                if reference.startswith("-l"):
                    candidates = {p for p in paths if p.name == "lib" + reference[2:] + ".a"
                                  or re.fullmatch(r"lib" + re.escape(reference[2:]) + r"\.so(?:\.[0-9]+)*", p.name)}
                    if len(candidates) != 1:
                        raise ValueError("ambiguous or missing recorded linker-script input")
                    candidate = next(iter(candidates))
                else:
                    candidate = _absolute(reference).resolve(strict=True)
                actual, magic = _file_identity(candidate)
                input_kind = _kind("link-input", actual["path"], magic)
                if input_kind not in {"elf", "static-archive"} or candidate not in paths:
                    raise ValueError("unsupported or unrecorded linker-script input")
                inputs.append(dict(actual, kind=input_kind, reference=reference))
            binding["inputs"] = inputs
        bindings[name] = binding
    return {"dependencies": bindings, "link_dependency_file": depfile}


def _bind(entry, metadata, observed, link_dependencies):
    result = validate_link_evidence(entry, metadata, link_dependencies)
    for name, identity in result["dependencies"].items():
        required = identity.get("inputs", []) if identity["kind"] == "linker-script" else [identity]
        for item in required:
            if item["kind"] == "elf" and observed.get(item["path"]) != item["sha256"]:
                raise ValueError("required ELF path/hash not observed: " + name)
    return result


def _observed_files(paths):
    observed = {}
    for path in paths:
        identity, magic = _file_identity(path)
        if not magic.startswith(b"\x7fELF"):
            raise ValueError("observed object lacks ELF magic")
        observed[identity["path"]] = identity["sha256"]
    return observed


def validate_artifact_resolution(entry: Mapping, metadata: Mapping, ldd_text: str,
                                 link_dependencies: Optional[Union[str, Path, Mapping]] = None):
    """Validate strict ldd resolution separately from child initialization."""
    if not isinstance(ldd_text, str) or "\x00" in ldd_text or len(ldd_text.encode("utf-8")) > 16 * 1024 * 1024:
        raise ValueError("invalid or oversized ldd evidence")
    paths, normalized = [], []
    for raw in ldd_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "not found" in line:
            raise ValueError("unresolved ldd dependency")
        no_address = re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", line)
        match = re.fullmatch(r"[^\s/]+\s+=>\s+(/[^\s]+)", no_address)
        if match:
            path = match[1]
        elif no_address.startswith("/") and not re.search(r"\s", no_address):
            path = no_address
        elif re.fullmatch(r"linux-vdso\.so\.\d+", no_address):
            normalized.append(no_address)
            continue
        else:
            raise ValueError("unrecognized ldd evidence")
        real = str(_absolute(path).resolve(strict=True))
        paths.append(real)
        normalized.append(no_address.replace(path, real))
    if not paths:
        raise ValueError("missing ldd ELF resolution evidence")
    observed = _observed_files(paths)
    return dict(_bind(entry, metadata, observed, link_dependencies),
                kind="ldd-resolution", normalized_lines=sorted(normalized),
                resolved_objects=observed)


def parse_ld_debug_trace(stderr: str, expected_program: str, expected_pid: Optional[int] = None):
    """Parse one child's observed glibc LD_DEBUG=libs events without IO.

    Non-loader stderr is retained separately. Unknown PID-prefixed loader syntax,
    extra processes, missing main events and ambiguous events reject. glibc does
    not emit a finalization event for every initialized object. Keep unmatched
    initialization observations, including objects loaded by destructors, without
    inventing finalizers or inferring process success or complete teardown.
    No path is inferred from search candidates or diagnostic text.
    """
    _absolute(expected_program)
    if expected_pid is not None and (type(expected_pid) is not int or expected_pid < 1):
        raise ValueError("invalid expected child PID")
    if not isinstance(stderr, str) or len(stderr.encode("utf-8")) > 16 * 1024 * 1024 or "\x00" in stderr:
        raise ValueError("invalid or oversized loader trace")
    pid, initialized, transferred, finished = None, False, False, False
    active, objects, finalized, events, application = set(), set(), set(), [], []
    for line in stderr.splitlines():
        match = re.fullmatch(r"\s*([1-9][0-9]*):\s*(.*)", line)
        if not match:
            if line.strip():
                if re.search(r"(?:calling (?:init|fini)|initialize program|transferring control):", line):
                    raise ValueError("loader event missing PID")
                application.append(line)
            continue
        current, message = int(match[1]), match[2].strip()
        if pid is None:
            pid = current
        if current != pid or (expected_pid is not None and current != expected_pid):
            raise ValueError("loader trace PID mismatch or multiple processes")
        if not message:
            continue
        program = re.fullmatch(r"(initialize program|transferring control):\s*(.+)", message)
        if program:
            if program[2] != expected_program or finished:
                raise ValueError("loader trace belongs to another program")
            if program[1] == "initialize program":
                if initialized or transferred:
                    raise ValueError("duplicate/out-of-order program initialization")
                initialized = True
            else:
                if not initialized or transferred:
                    raise ValueError("duplicate/out-of-order transfer")
                transferred = True
            events.append({"event": program[1], "path": program[2]})
            continue
        init = re.fullmatch(r"calling init:\s*(/.+?)(?:\s+\[0\])?", message)
        fini = re.fullmatch(r"calling fini:\s*(.*?)\s*\[0\]", message)
        if init:
            path = init[1]
            _absolute(path)
            if path in active:
                raise ValueError("duplicate/out-of-order object initialization")
            active.add(path)
            objects.add(path)
            events.append({"event": "calling init", "path": path})
        elif fini:
            path = fini[1].strip()
            if not path or path == expected_program:
                if not transferred or finished:
                    raise ValueError("duplicate/out-of-order main finalization")
                finished = True
            else:
                _absolute(path)
                if path not in active:
                    raise ValueError("object finalization has no initialization")
                active.remove(path)
                finalized.add(path)
            events.append({"event": "calling fini", "path": path})
        elif (re.fullmatch(r"find library=.+ \[0\]; searching", message)
              or re.fullmatch(r"search path=.+\(.+\)", message)
              or re.fullmatch(r"search cache=/[^\s]+", message)
              or re.fullmatch(r"trying file=/[^\s]+", message)):
            pass
        else:
            raise ValueError("unrecognized loader trace record: " + message[:160])
    if not initialized or not transferred or not finished or not objects:
        raise ValueError("missing required main loader events or object initialization")
    return {"kind": "ld-debug-libs-initialization-events", "pid": pid,
            "program": expected_program, "initialized_paths": sorted(objects),
            "observed_finalization_paths": sorted(finalized),
            "initializations_without_observed_finalization": sorted(active),
            "main_finalization_observed": finished,
            "events": events, "application_stderr": "\n".join(application),
            "trace_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "scope": "observed loader events; absent object finalization is not a failure or proof of teardown; process exit is checked separately"}


def validate_loaded_trace(entry: Mapping, metadata: Mapping, trace: str,
                          expected_program: str, expected_pid: Optional[int] = None,
                          link_dependencies: Optional[Union[str, Path, Mapping]] = None):
    """Bind diagnostic events to this immutable executable and provider files.

    The harness owns child argv/env/PID/exit/stdout/stderr capture and successful
    numerical-result validation. This function alone does not attest execution
    success, symbol ownership, all process mappings, or effective worker counts.
    """
    if expected_program != entry.get("executable_path"):
        raise ValueError("expected program differs from artifact")
    program, magic = _file_identity(expected_program)
    if (program["path"] != expected_program or not magic.startswith(b"\x7fELF")
            or program["sha256"] != entry.get("binary_sha256")):
        raise ValueError("traced artifact binary identity differs")
    parsed = parse_ld_debug_trace(trace, expected_program, expected_pid)
    observed = _observed_files(parsed["initialized_paths"])
    return dict(_bind(entry, metadata, observed, link_dependencies),
                kind=parsed["kind"], process=parsed, binary=program,
                initialized_objects=[{"path": path, "sha256": digest}
                                     for path, digest in sorted(observed.items())],
                scope="observed glibc initialization events; not symbol ownership or worker count")
