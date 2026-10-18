#!/usr/bin/env bash
# Scheduler-neutral body; the reviewed packet supplies PBS directives and values.
set -uo pipefail
umask 077
fail() { printf 'AMD allocated build: %s\n' "$*" >&2; exit 2; }
[[ ${GPU_SUITE_SIRIUS_AMD_BUILD_APPROVED:-} == yes && -n ${PBS_JOBID:-} ]] || fail 'approved PBS allocation required'
[[ ${PBS_ENVIRONMENT:-PBS_BATCH} == PBS_BATCH ]] || fail 'PBS batch allocation required'
for name in AMD_BUILD_SOURCE_ROOT AMD_BUILD_RUN_ROOT AMD_BUILD_RUN_ID AMD_BUILD_SOURCE_SHA256 AMD_BUILD_GPU_ARCH AMD_BUILD_PYTHON AMD_BUILD_TIMEOUT_SECONDS AMD_BUILD_CONTROLLER_SHA256 AMD_BUILD_MODULES_SHA256; do
    [[ -n ${!name:-} ]] || fail "missing $name"
    export "$name=${!name}"
done
[[ $AMD_BUILD_SOURCE_ROOT == /* && $AMD_BUILD_RUN_ROOT == "$AMD_BUILD_SOURCE_ROOT/runs/$AMD_BUILD_RUN_ID" ]] || fail 'invalid source/run paths'
[[ $AMD_BUILD_RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $AMD_BUILD_RUN_ID != *..* ]] || fail 'invalid run ID'
[[ $AMD_BUILD_TIMEOUT_SECONDS =~ ^[0-9]+$ && $AMD_BUILD_TIMEOUT_SECONDS -ge 30 && $AMD_BUILD_TIMEOUT_SECONDS -le 86400 ]] || fail 'invalid timeout'
[[ -d $AMD_BUILD_SOURCE_ROOT && -d $AMD_BUILD_RUN_ROOT && ! -L $AMD_BUILD_SOURCE_ROOT && ! -L $AMD_BUILD_RUN_ROOT && ! -L $AMD_BUILD_SOURCE_ROOT/runs ]] || fail 'real directories required'
for path in "$AMD_BUILD_SOURCE_ROOT" "$AMD_BUILD_SOURCE_ROOT/runs" "$AMD_BUILD_RUN_ROOT"; do
    [[ $(cd -- "$path" && pwd -P) == "$path" ]] || fail 'canonical directories required'
done
[[ $AMD_BUILD_SOURCE_SHA256 =~ ^[0-9a-f]{64}$ ]] || fail 'invalid source snapshot hash'
[[ $AMD_BUILD_GPU_ARCH =~ ^gfx[0-9a-z]+(:[a-z0-9]+[+-])*$ ]] || fail 'invalid explicit GPU target'
[[ $AMD_BUILD_PYTHON == /* && -f $AMD_BUILD_PYTHON && -x $AMD_BUILD_PYTHON ]] || fail 'explicit Python executable required'
export AMD_BUILD_CPU_SPARSE_BACKEND=${AMD_BUILD_CPU_SPARSE_BACKEND:-ONEMKL}
[[ $AMD_BUILD_CPU_SPARSE_BACKEND == ONEMKL || $AMD_BUILD_CPU_SPARSE_BACKEND == AOCLSPARSE ]] || fail 'invalid explicit CPU Sparse provider'
export AMD_BUILD_MODE=${AMD_BUILD_MODE:-all}
case $AMD_BUILD_MODE in
    all) export SIRIUS_COMPUTE_MPI_MODULE='' ;;
    clean-benchmarks)
        [[ ${AMD_BUILD_EXPECTED_COMMIT:-} =~ ^[0-9a-f]{40}$ ]] || fail 'approved clean commit required'
        [[ ${AMD_BUILD_ROCM_VERSION_FILE:-} == /* && -f $AMD_BUILD_ROCM_VERSION_FILE ]] || fail 'explicit ROCm version file required'
        export AMD_BUILD_EXPECTED_COMMIT AMD_BUILD_ROCM_VERSION_FILE
        export SIRIUS_COMPUTE_MPI_MODULE=openmpi/5.0.10/rocm7.2.0
        ;;
    *) fail 'unknown AMD_BUILD_MODE' ;;
esac
command -v timeout >/dev/null 2>&1 || fail 'timeout unavailable'
check_hash() {
    local path=$1 expected=$2 actual
    [[ $expected =~ ^[0-9a-f]{64}$ && -f $path && ! -L $path ]] || return 1
    actual=$(sha256sum "$path") || return 1
    [[ ${actual%% *} == "$expected" ]]
}
check_hash "$AMD_BUILD_RUN_ROOT/amd_build.py" "$AMD_BUILD_CONTROLLER_SHA256" || fail 'controller hash mismatch'
check_hash "$AMD_BUILD_RUN_ROOT/rocfft_compute_modules.sh" "$AMD_BUILD_MODULES_SHA256" || fail 'module helper hash mismatch'
# Legacy names adapt the source-only module helper; MPI is opt-in above.
export STAGE31_SOURCE_ROOT=$AMD_BUILD_SOURCE_ROOT STAGE31_RUN_ROOT=$AMD_BUILD_RUN_ROOT STAGE31_RUN_ID=$AMD_BUILD_RUN_ID
export GPU_SUITE_SIRIUS_COMPUTE_APPROVED=yes PYTHONDONTWRITEBYTECODE=1
set -C
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$AMD_BUILD_RUN_ROOT/amd-build-job-start.txt" || fail 'existing attempt; no retry'
exec 3> "$AMD_BUILD_RUN_ROOT/amd-build-job.stdout" || fail 'existing stdout'
exec 4> "$AMD_BUILD_RUN_ROOT/amd-build-job.stderr" || fail 'existing stderr'
printf 'USER=%s HOSTNAME=%s HOME=%s PATH=%s PBS_JOBID=%s\n' "$(id -un)" "$(hostname)" "${HOME:-}" "$PATH" "$PBS_JOBID" >&3
unset BASH_ENV ENV
# Login startup provides the site module function. No owner terminal is changed.
# shellcheck disable=SC2016
TERM=dumb timeout --signal=TERM --kill-after=15 "$((AMD_BUILD_TIMEOUT_SECONDS + 60))" /bin/bash --login -i -c '
source "$1" || exit $?
shift
extra=()
if [[ $AMD_BUILD_MODE == clean-benchmarks ]]; then
    extra=(--clean-benchmarks --expected-git-commit "$AMD_BUILD_EXPECTED_COMMIT" --rocm-version-file "$AMD_BUILD_ROCM_VERSION_FILE")
fi
exec "$AMD_BUILD_PYTHON" -B "$1" \
  --source-root "$AMD_BUILD_SOURCE_ROOT" --run-root "$AMD_BUILD_RUN_ROOT" \
  --run-id "$AMD_BUILD_RUN_ID" --expected-source-sha256 "$AMD_BUILD_SOURCE_SHA256" \
  --gpu-arch "$AMD_BUILD_GPU_ARCH" --timeout-seconds "$AMD_BUILD_TIMEOUT_SECONDS" \
  --cpu-c-compiler "$CPU_CC" --cpu-cxx-compiler "$CPU_CXX" \
  --amd-cxx-compiler "$HIP_CLANG" --rocm-root "$ROCM_PATH" \
  --cpu-blas-backend GENERIC_CBLAS --cpu-lapack-backend GENERIC_LAPACKE \
  --cpu-sparse-backend "$AMD_BUILD_CPU_SPARSE_BACKEND" \
  --aocl-cpu-candidates "${extra[@]}"
' bash "$AMD_BUILD_RUN_ROOT/rocfft_compute_modules.sh" "$AMD_BUILD_RUN_ROOT/amd_build.py" </dev/null >&3 2>&4
rc=$?
printf '%s\n' "$rc" > "$AMD_BUILD_RUN_ROOT/amd-build-job-exit.txt" || exit 2
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$AMD_BUILD_RUN_ROOT/amd-build-job-end.txt" || exit 2
printf 'GPU_SUITE_SIRIUS_AMD_BUILD_END %s %s\n' "$AMD_BUILD_RUN_ID" "$rc"
exit "$rc"
