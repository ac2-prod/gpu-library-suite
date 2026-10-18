#!/usr/bin/env bash
# Run only after separate approval, inside an allocated Sirius compute node.
# This body neither obtains an allocation nor starts a network connection.
set -euo pipefail
umask 077

fail() { printf 'rocFFT smoke: %s\n' "$*" >&2; exit 2; }
[[ ${GPU_SUITE_ALLOCATED_COMPUTE_NODE:-} == yes ]] || fail 'explicit allocated-compute-node confirmation is required; never run on a login node'
for variable in SOURCE_ROOT RUN_ROOT CMAKE PYTHON CPU_CC CPU_CXX HIP_CLANG OPENMP_CXX ROCM_ROOT FFTW_ROOT GPU_ARCH BUILD_JOBS HIP_DEVICE OMP_DEVICE; do
    [[ -n ${!variable:-} ]] || fail "missing owner/site input: $variable"
done
[[ $BUILD_JOBS =~ ^[1-9][0-9]*$ ]] || fail 'BUILD_JOBS must be a positive approved count'
[[ $HIP_DEVICE =~ ^[0-9]+$ && $OMP_DEVICE =~ ^[0-9]+$ ]] || fail 'HIP_DEVICE and OMP_DEVICE must be independent nonnegative runtime indices'
[[ $GPU_ARCH != *native* && $GPU_ARCH != *';'* && $GPU_ARCH != *' '* ]] || fail 'GPU_ARCH must be one explicit target, never native'
[[ ${HIP_CLANG##*/} != hipcc ]] || fail 'native CMake HIP requires supported Clang, not hipcc'
for variable in CMAKE PYTHON CPU_CC CPU_CXX HIP_CLANG OPENMP_CXX; do
    executable=${!variable}
    [[ $executable == /* && -f $executable && -x $executable ]] || fail "$variable must name an executable absolute path"
done
[[ $SOURCE_ROOT == /* && $RUN_ROOT == /* && $ROCM_ROOT == /* && $FFTW_ROOT == /* ]] || fail 'source, output and dependency roots must be absolute paths'
[[ -d $SOURCE_ROOT && -d $ROCM_ROOT && -d $FFTW_ROOT ]] || fail 'source and confirmed dependency directories must exist'
[[ ! -e $RUN_ROOT && ! -L $RUN_ROOT ]] || fail 'RUN_ROOT must not already exist'
SMOKE_CONFIG=${SMOKE_CONFIG:-"$SOURCE_ROOT/jobs/sirius/rocfft-smoke.json"}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SOURCE_ROOT/tools${PYTHONPATH:+:$PYTHONPATH}"
# Validate the requested output and workload before creating files or compiling.
"$PYTHON" - "$SOURCE_ROOT" "$RUN_ROOT" "$SMOKE_CONFIG" <<'PY'
import sys
from pathlib import Path
from gpu_suite.config import load_config
source, output, config = (Path(value).resolve() for value in sys.argv[1:])
doc = load_config(config)
if doc.get('vendor') != 'amd' or set(doc['benchmarks']) != {'rocfft'} or doc['run_mode'] != 'smoke':
    raise SystemExit('only an AMD rocFFT smoke configuration is accepted')
if not doc['benchmarks']['rocfft']['enabled']:
    raise SystemExit('rocFFT must be enabled in the smoke configuration')
if doc['continue_on_failure']:
    raise SystemExit('the bounded smoke must stop on first failure')
cases = doc['benchmarks']['rocfft']['cases']
if not cases or any(case['parameters']['batch'] < 2 for case in cases):
    raise SystemExit('every smoke case must exercise multiple batches')
if any(set(case['scopes']) != {'compute', 'end-to-end'} for case in cases):
    raise SystemExit('both timing scopes are required')
# This is a bounded correctness smoke, not a production workload.
if any(case['parameters']['nfft'] > 256 or case['parameters']['batch'] > 8
       or any(scope['repeat'] > 2 or scope['trials'] > 2 or scope['warmup'] > 1
              for scope in case['scopes'].values()) for case in cases):
    raise SystemExit('smoke dimensions/counts exceed the reviewed small bounds')
sys.path.insert(0, str(source / 'jobs/sirius'))
from rocfft_compute_smoke import validate_run_location
validate_run_location(Path(sys.argv[1]), Path(sys.argv[2]))
if not output.parent.is_dir():
    raise SystemExit('runs/ parent must already exist')
PY
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs"
mkdir "$RUN_ROOT/tmp" "$RUN_ROOT/cache"
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUN_ROOT/cache"
export ROCFFT_RTC_CACHE_PATH="$RUN_ROOT/cache/rocfft.db"
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_DEFAULT_DEVICE="$OMP_DEVICE"
export GIT_OPTIONAL_LOCKS=0

step=0
LAST_STDOUT=
run() {
    local name=$1 rc
    shift
    step=$((step + 1))
    local prefix
    prefix="$RUN_ROOT/logs/$(printf '%03d' "$step")-$name"
    LAST_STDOUT="$prefix.stdout"
    printf '%q ' "$@" > "$prefix.argv"
    printf '\n' >> "$prefix.argv"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$prefix.started"
    if "$@" > "$prefix.stdout" 2> "$prefix.stderr"; then rc=0; else rc=$?; fi
    printf '%s\n' "$rc" > "$prefix.exit"
    date -u '+%Y-%m-%dT%H:%M:%SZ' > "$prefix.ended"
    if (( rc != 0 )); then
        printf 'Step %s failed (exit %s); inspect %s.*\n' "$name" "$rc" "$prefix" >&2
    fi
    return "$rc"
}
finish() { local rc=$?; printf '%s\n' "$rc" > "$RUN_ROOT/smoke-exit.txt"; }
trap finish EXIT
run cmake-version "$CMAKE" --version
run python-version "$PYTHON" --version
run cpu-c-version "$CPU_CC" --version
run cpu-cxx-version "$CPU_CXX" --version
run hip-clang-version "$HIP_CLANG" --version
run openmp-cxx-version "$OPENMP_CXX" --version
run source-snapshot "$PYTHON" - "$SOURCE_ROOT" <<'PY'
import sys
from pathlib import Path
from hash_source_snapshot import git_source_paths
from gpu_suite.hashing import source_snapshot_sha256
source = Path(sys.argv[1])
sys.path.insert(0, str(source / 'jobs/sirius'))
from rocfft_compute_smoke import snapshot_paths
print(source_snapshot_sha256(source, snapshot_paths(git_source_paths(source))))
PY
SOURCE_HASH=$(cat "$LAST_STDOUT")

common=(-S "$SOURCE_ROOT" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    -DGPU_SUITE_VENDOR=amd -DGPU_SUITE_AMD_LIBRARIES=rocfft -DGPU_SUITE_SOURCE_LANGUAGE=c-cpp
    -DGPU_SUITE_BUILD_CUDA=OFF -DGPU_SUITE_BUILD_OPENACC=OFF
    -DGPU_SUITE_BUILD_TESTS=OFF -DPython3_EXECUTABLE="$PYTHON"
    -DCMAKE_C_COMPILER="$CPU_CC" -DCMAKE_CXX_COMPILER="$CPU_CXX"
    -DGPU_SUITE_ROCM_ROOT="$ROCM_ROOT" -DFFTW_ROOT="$FFTW_ROOT")
run configure-cpu "$CMAKE" "${common[@]}" -B "$RUN_ROOT/cpu" \
    -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_HIP=OFF -DGPU_SUITE_BUILD_OPENMP=OFF
run configure-hip "$CMAKE" "${common[@]}" -B "$RUN_ROOT/hip" \
    -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_HIP=ON -DGPU_SUITE_BUILD_OPENMP=OFF \
    -DCMAKE_HIP_COMPILER="$HIP_CLANG" -DCMAKE_HIP_ARCHITECTURES="$GPU_ARCH"
run configure-openmp "$CMAKE" "${common[@]}" -B "$RUN_ROOT/openmp" \
    -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_HIP=OFF -DGPU_SUITE_BUILD_OPENMP=ON \
    -DCMAKE_CXX_COMPILER="$OPENMP_CXX" -DGPU_SUITE_AMD_GPU_TARGET="$GPU_ARCH"
run build-cpu "$CMAKE" --build "$RUN_ROOT/cpu" --parallel "$BUILD_JOBS" --verbose --target fft_cpu fft_cpu_bench gpu_suite_partial_manifest
run build-hip "$CMAKE" --build "$RUN_ROOT/hip" --parallel "$BUILD_JOBS" --verbose --target fft_gpu fft_gpu_bench gpu_suite_partial_manifest
run build-openmp "$CMAKE" --build "$RUN_ROOT/openmp" --parallel "$BUILD_JOBS" --verbose --target openmp_rocfft openmp_rocfft_bench gpu_suite_partial_manifest
run merge-manifests "$PYTHON" "$SOURCE_ROOT/tools/merge_manifests.py" \
    --output "$RUN_ROOT/executables.json" "$RUN_ROOT/cpu/partial-manifest.json" \
    "$RUN_ROOT/hip/partial-manifest.json" "$RUN_ROOT/openmp/partial-manifest.json"
# ldd observes actual resolution in this allocation and the current runtime path.
# It does not assert that a CMake header-only check proved linkage.
for profile in cpu hip openmp; do
    case "$profile" in
        cpu) executable=fft_cpu_bench ;;
        hip) executable=fft_gpu_bench ;;
        openmp) executable=openmp_rocfft_bench ;;
    esac
    run "resolved-$profile" ldd "$RUN_ROOT/$profile/amd/c-cpp/rocfft/$executable"
done
run prepare-evidence "$PYTHON" "$SOURCE_ROOT/jobs/sirius/rocfft_smoke_evidence.py" \
    "$RUN_ROOT" "$SMOKE_CONFIG" "$HIP_DEVICE" "$OMP_DEVICE"
RUNTIME_HASH=$(cat "$LAST_STDOUT")
[[ $SOURCE_HASH =~ ^[0-9a-f]{64}$ && $RUNTIME_HASH =~ ^[0-9a-f]{64}$ ]] || fail 'missing source/runtime hash evidence'
for format in jsonl csv; do
    # The first small case also proves actual HIP/OpenMP pointer interoperation.
    run "runner-$format" "$PYTHON" "$SOURCE_ROOT/tools/run_suite.py" \
        --config "$RUN_ROOT/config-$format.json" --manifest "$RUN_ROOT/executables.json" \
        --build-metadata "$RUN_ROOT/cpu/build-metadata.json" \
        --build-metadata "$RUN_ROOT/hip/build-metadata.json" \
        --build-metadata "$RUN_ROOT/openmp/build-metadata.json" \
        --run-id "rocfft-smoke-$format" --system-label Sirius --wave 0 --node-index 0 \
        --device "$HIP_DEVICE" --source-snapshot-sha256 "$SOURCE_HASH" \
        --runtime-environment-sha256 "$RUNTIME_HASH" --output "$RUN_ROOT/raw.$format"
    run "validate-$format" "$PYTHON" "$SOURCE_ROOT/tools/validate_results.py" \
        --config "$RUN_ROOT/config-$format.json" --manifest "$RUN_ROOT/executables.json" \
        --report-output "$RUN_ROOT/validation-$format.json" "$RUN_ROOT/raw.$format"
done
printf 'rocFFT smoke completed; inspect %s and its per-command logs.\n' "$RUN_ROOT"
