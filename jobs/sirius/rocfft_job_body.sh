#!/usr/bin/env bash
# Scheduler-neutral body. The reviewed dispatcher supplies PBS directives.
# This body neither submits jobs nor changes modules or GPU visibility.
set -uo pipefail
umask 077

run_id=${STAGE31_RUN_ID:-unknown}
cpu_threads=${STAGE31_CPU_THREADS-1}
include_serial=${STAGE31_INCLUDE_SERIAL-no}
child_pid=
stdout_path=
stderr_path=
finished=0

fail() {
    printf 'GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL %s %s %s\n' "$run_id" "$1" "$2"
    exit "$2"
}

finish() {
    local rc=$1 fallback=$2 line markers=0 marker_status='' stage='' status='' rest='' marker_line=''
    (( finished == 0 )) || return
    finished=1
    trap - TERM INT HUP
    # The helper normally emits exactly one marker; re-emit it only once.
    if [[ -f $stdout_path ]]; then
        while IFS= read -r line || [[ -n $line ]]; do
            # Every marker-prefixed line participates in uniqueness, including
            # malformed lines and another run's marker. Keep originals in the
            # process log, but never let them accompany a successful result.
            if [[ $line == 'GPU_SUITE_SIRIUS_ROCFFT_SMOKE'* ]]; then
                markers=$((markers + 1)); marker_status=''; marker_line=''
            else
                printf '%s\n' "$line"
                continue
            fi
            if [[ $line == "GPU_SUITE_SIRIUS_ROCFFT_SMOKE PASS $run_id" ]]; then
                marker_status=PASS; marker_line=$line
            elif [[ $line == "GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL $run_id "* ]]; then
                rest=${line#"GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL $run_id "}
                stage=${rest%% *}; status=${rest#* }
                if [[ $stage =~ ^[A-Za-z0-9_-]+$ && $status =~ ^-?[0-9]+$ ]]; then
                    marker_status=FAIL; marker_line=$line
                fi
            fi
        done < "$stdout_path"
    fi
    [[ ! -f $stderr_path ]] || cat -- "$stderr_path" >&2
    if (( markers == 1 )) && { [[ $marker_status == PASS && $rc == 0 ]] || [[ $marker_status == FAIL && $rc != 0 ]]; }; then
        printf '%s\n' "$marker_line"
        exit "$rc"
    fi
    # Missing/inconsistent completion is a failure, never an implicit success.
    (( rc != 0 )) || rc=1
    printf 'GPU_SUITE_SIRIUS_ROCFFT_SMOKE FAIL %s %s %s\n' "$run_id" "$fallback" "$rc"
    exit "$rc"
}

interrupted() {
    local rc=$1 stage=$2
    trap '' TERM INT HUP
    if [[ -n $child_pid ]]; then
        # Only the Python child launched by this body. PBS owns job cleanup.
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    finish "$rc" "$stage"
}

[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $run_id != *..* ]] || fail INVALID_RUN_ID 2
[[ $cpu_threads =~ ^([1-9]|1[0-9]|2[0-4])$ ]] || fail CPU_THREADS_EXCEED_ALLOCATION 2
[[ $include_serial == yes || $include_serial == no ]] || fail INCLUDE_SERIAL_VALUE 2
[[ $include_serial != yes || $cpu_threads -gt 1 ]] || fail INCLUDE_SERIAL_REQUIRES_THREADS 2
campaign_args=(--cpu-threads "$cpu_threads")
[[ $include_serial != yes ]] || campaign_args+=(--include-serial-campaign)
for variable in STAGE31_SOURCE_ROOT STAGE31_RUN_ROOT STAGE31_PYTHON PBS_JOBID; do
    [[ -n ${!variable:-} ]] || fail "MISSING_$variable" 2
done
[[ $STAGE31_SOURCE_ROOT == /* && -d $STAGE31_SOURCE_ROOT && ! -L $STAGE31_SOURCE_ROOT ]] || fail SOURCE_ROOT 2
[[ $STAGE31_RUN_ROOT == /* && -d $STAGE31_RUN_ROOT && ! -L $STAGE31_RUN_ROOT ]] || fail RUN_ROOT 2
[[ $STAGE31_PYTHON == /* && -f $STAGE31_PYTHON && -x $STAGE31_PYTHON ]] || fail PYTHON_EXECUTABLE 2
source_real=$(cd -- "$STAGE31_SOURCE_ROOT" && pwd -P) || fail SOURCE_ROOT 2
run_real=$(cd -- "$STAGE31_RUN_ROOT" && pwd -P) || fail RUN_ROOT 2
# The approved output is exactly runs/<RUN_ID> below the source checkout.
# Reject symlinks and noncanonical spellings before opening any log files.
[[ $STAGE31_SOURCE_ROOT == "$source_real" && $STAGE31_RUN_ROOT == "$run_real" ]] || fail NONCANONICAL_ROOT 2
[[ ! -L $source_real/runs && $run_real == "$source_real/runs/$run_id" ]] || fail RUN_ROOT_LAYOUT 2
[[ -f $run_real/compute_smoke.py && ! -L $run_real/compute_smoke.py ]] || fail COMPUTE_PAYLOAD 2

printf 'COMPUTE_ENVIRONMENT USER=%s HOSTNAME=%s HOME=%s PATH=%s\n' \
    "$(id -un)" "${HOSTNAME:-unavailable}" "${HOME:-unavailable}" "${PATH:-unavailable}"
printf 'STAGE31_JOB_START RUN_ID=%s PBS_JOBID=%s HOSTNAME=%s\n' "$run_id" "$PBS_JOBID" "${HOSTNAME:-unavailable}"
date -u '+UTC=%Y-%m-%dT%H:%M:%SZ' || fail UTC_CLOCK 2
printf 'SITE_CONFIRMED_ALLOCATION select=1 ncpus=24 ngpus=1 CPU_THREADS=%s BUILD_PARALLELISM=1\n' "$cpu_threads"
if command -v taskset >/dev/null 2>&1; then
    taskset -pc "$$" || printf 'CPU_AFFINITY_UNAVAILABLE\n'
else
    printf 'CPU_AFFINITY_UNAVAILABLE\n'
fi

stdout_path=$run_real/compute-process.stdout
stderr_path=$run_real/compute-process.stderr
[[ ! -e $stdout_path && ! -L $stdout_path && ! -e $stderr_path && ! -L $stderr_path ]] || fail OUTPUT_ALREADY_EXISTS 2
set -C
exec 3> "$stdout_path" || fail OUTPUT_ALREADY_EXISTS 2
exec 4> "$stderr_path" || fail OUTPUT_ALREADY_EXISTS 2
export GPU_SUITE_ALLOCATED_COMPUTE_NODE=yes GPU_SUITE_SITE_BINDING_VERIFIED=yes
export PYTHONDONTWRITEBYTECODE=1
trap 'interrupted 143 TERMINATED' TERM
trap 'interrupted 130 INTERRUPTED' INT
trap 'interrupted 129 HANGUP' HUP
"$STAGE31_PYTHON" -B "$run_real/compute_smoke.py" \
    --source-root "$source_real" --run-root "$run_real" --run-id "$run_id" \
    --scheduler PBS --job-id "$PBS_JOBID" --allocated-cpus 24 --allocated-gpus 1 \
    "${campaign_args[@]}" \
    >&3 2>&4 &
child_pid=$!
if wait "$child_pid"; then rc=0; else rc=$?; fi
child_pid=
exec 3>&- 4>&-
finish "$rc" HELPER_COMPLETION
