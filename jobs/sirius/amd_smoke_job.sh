#!/usr/bin/env bash
# Allocated-job body only. The reviewed packet and PBS wrapper own resources.
set -uo pipefail
umask 077
fail() { printf 'AMD allocated smoke: %s\n' "$*" >&2; exit 2; }

[[ ${GPU_SUITE_SIRIUS_AMD_SMOKE_APPROVED:-} == yes &&
   ${PBS_JOBID:-} =~ ^[0-9]+\.[A-Za-z0-9_.-]+$ &&
   ${PBS_ENVIRONMENT:-} == PBS_BATCH ]] || fail 'approved PBS batch allocation required'
for name in AMD_SMOKE_SOURCE_ROOT AMD_SMOKE_RUN_ROOT AMD_SMOKE_RUN_ID AMD_SMOKE_PYTHON AMD_SMOKE_TIMEOUT_SECONDS AMD_SMOKE_PACKET_SHA256 AMD_SMOKE_CONTROLLER_SHA256 AMD_SMOKE_MODULES_SHA256; do
    [[ -n ${!name:-} ]] || fail "missing $name"
    export "$name=${!name}"
done
[[ $AMD_SMOKE_RUN_ID =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ && $AMD_SMOKE_RUN_ID != *..* ]] || fail 'invalid run ID'
[[ $AMD_SMOKE_SOURCE_ROOT == /* && $AMD_SMOKE_RUN_ROOT == "$AMD_SMOKE_SOURCE_ROOT/runs/$AMD_SMOKE_RUN_ID" ]] || fail 'invalid source/run paths'
[[ $AMD_SMOKE_TIMEOUT_SECONDS =~ ^[1-9][0-9]{1,3}$ &&
   $AMD_SMOKE_TIMEOUT_SECONDS -ge 30 && $AMD_SMOKE_TIMEOUT_SECONDS -le 1800 ]] || fail 'invalid timeout'
for path in "$AMD_SMOKE_SOURCE_ROOT" "$AMD_SMOKE_SOURCE_ROOT/runs" "$AMD_SMOKE_RUN_ROOT"; do
    [[ -d $path && ! -L $path ]] || fail 'real directories required'
    [[ $(cd -- "$path" && pwd -P) == "$path" ]] || fail 'canonical directories required'
done
[[ $AMD_SMOKE_PYTHON == /* && -f $AMD_SMOKE_PYTHON && -x $AMD_SMOKE_PYTHON ]] || fail 'explicit Python executable required'
command -v timeout >/dev/null 2>&1 || fail 'timeout unavailable'
command -v sha256sum >/dev/null 2>&1 || fail 'sha256sum unavailable'
check_hash() {
    local path=$1 expected=$2 actual
    [[ $expected =~ ^[0-9a-f]{64}$ && -f $path && ! -L $path ]] || return 1
    actual=$(sha256sum "$path") || return 1
    [[ ${actual%% *} == "$expected" ]]
}
check_hash "$AMD_SMOKE_RUN_ROOT/packet.json" "$AMD_SMOKE_PACKET_SHA256" || fail 'packet hash mismatch'
check_hash "$AMD_SMOKE_RUN_ROOT/amd_smoke.py" "$AMD_SMOKE_CONTROLLER_SHA256" || fail 'controller hash mismatch'
check_hash "$AMD_SMOKE_RUN_ROOT/rocfft_compute_modules.sh" "$AMD_SMOKE_MODULES_SHA256" || fail 'module helper hash mismatch'

# Never truncate earlier attempt logs, including incomplete attempts.
for name in amd-smoke-job-start.txt amd-smoke-job-end.txt amd-smoke-job-exit.txt amd-smoke-job.stdout amd-smoke-job.stderr; do
    [[ ! -e $AMD_SMOKE_RUN_ROOT/$name && ! -L $AMD_SMOKE_RUN_ROOT/$name ]] || fail 'existing attempt; no retry'
done
[[ ! -e $AMD_SMOKE_RUN_ROOT/smoke && ! -L $AMD_SMOKE_RUN_ROOT/smoke ]] || fail 'existing controller attempt; no retry'
set -C
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$AMD_SMOKE_RUN_ROOT/amd-smoke-job-start.txt" || fail 'cannot reserve new attempt'
exec 3> "$AMD_SMOKE_RUN_ROOT/amd-smoke-job.stdout" || fail 'cannot reserve stdout'
exec 4> "$AMD_SMOKE_RUN_ROOT/amd-smoke-job.stderr" || fail 'cannot reserve stderr'
printf 'USER=%s HOSTNAME=%s HOME=%s PATH=%s PBS_JOBID=%s\n' "$(id -un)" "$(hostname)" "${HOME:-}" "$PATH" "$PBS_JOBID" >&3

# Adapt only to the unchanged source-only helper in this disposable job shell.
export STAGE31_SOURCE_ROOT=$AMD_SMOKE_SOURCE_ROOT STAGE31_RUN_ROOT=$AMD_SMOKE_RUN_ROOT STAGE31_RUN_ID=$AMD_SMOKE_RUN_ID
export GPU_SUITE_SIRIUS_COMPUTE_APPROVED=yes PYTHONDONTWRITEBYTECODE=1
unset BASH_ENV ENV
# The controller independently verifies its packet, helpers, source and binaries
# before invoking any benchmark. timeout owns only this child process group.
# shellcheck disable=SC2016
TERM=dumb timeout --signal=TERM --kill-after=15 "$AMD_SMOKE_TIMEOUT_SECONDS" /bin/bash --login -i -c '
source "$1" || exit $?
shift
exec "$AMD_SMOKE_PYTHON" -B "$1" \
  --packet "$AMD_SMOKE_RUN_ROOT/packet.json" \
  --expected-packet-sha256 "$AMD_SMOKE_PACKET_SHA256"
' bash "$AMD_SMOKE_RUN_ROOT/rocfft_compute_modules.sh" "$AMD_SMOKE_RUN_ROOT/amd_smoke.py" </dev/null >&3 2>&4
rc=$?
printf '%s\n' "$rc" > "$AMD_SMOKE_RUN_ROOT/amd-smoke-job-exit.txt" || exit 2
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$AMD_SMOKE_RUN_ROOT/amd-smoke-job-end.txt" || exit 2
printf 'GPU_SUITE_SIRIUS_AMD_SMOKE_END %s %s\n' "$AMD_SMOKE_RUN_ID" "$rc"
exit "$rc"
