#!/usr/bin/env bash
# Run in an owner-authenticated interactive login shell. Build only, then stop.
set -euo pipefail
umask 077

fail() { printf 'Sirius module build: %s\n' "$*" >&2; exit 2; }
[[ ${GPU_SUITE_SIRIUS_BUILD_ONLY_APPROVED:-} == yes ]] || fail 'build-only approval is required'
[[ $# == 3 ]] || fail 'expected SOURCE_ROOT RUN_ROOT RUN_ID'
source_root=$1
run_root=$2
run_id=$3
[[ $source_root == /* && $run_root == "$source_root/runs/$run_id" ]] || fail 'invalid source/run paths'
[[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $run_id != *..* ]] || fail 'invalid run ID'
[[ -d $source_root && -d $run_root && ! -L $source_root && ! -L $run_root && ! -L $source_root/runs ]] || fail 'expected real source/run directories'
command -v module >/dev/null 2>&1 || fail 'module unavailable in interactive login shell'
export PAGER=cat MODULES_PAGER=cat LMOD_PAGER=cat PYTHONDONTWRITEBYTECODE=1

printf 'BUILD_ONLY_USER=%s HOSTNAME=%s HOME=%s\n' "$(id -un)" "$(hostname)" "${HOME:-}"
date -u '+BUILD_ONLY_UTC=%Y-%m-%dT%H:%M:%SZ'
printf 'INITIAL_MODULES\n'
module -t list

# A wholly absent/empty variable has no entries. Empty colon-separated entries
# would search the current directory; relative entries are also unsuitable here.
validate_library_path() {
    local label=$1 remaining=$2 entry
    [[ -n $remaining ]] || return 0
    while :; do
        entry=${remaining%%:*}
        [[ $entry == /* ]] || fail "$label contains an empty or relative LD_LIBRARY_PATH entry"
        [[ $remaining == *:* ]] || break
        remaining=${remaining#*:}
    done
}

prioritize_gcc_libraries() {
    local before=$1 after=$2 loaded=$3 remaining entry
    gcc_library_path=
    remaining=$after
    while [[ -n $remaining ]]; do
        entry=${remaining%%:*}
        if [[ :$before: != *":$entry:"* && :$gcc_library_path: != *":$entry:"* ]]; then
            [[ -d $entry ]] || fail 'GCC module introduced a missing library directory'
            gcc_library_path=${gcc_library_path:+$gcc_library_path:}$entry
        fi
        [[ $remaining == *:* ]] || break
        remaining=${remaining#*:}
    done
    [[ -n $gcc_library_path ]] || fail 'GCC module introduced no usable library directories'
    # Preserve module-provided order and all AOCL/unrelated entries, comparing
    # whole path entries literally rather than substring or pattern matches.
    effective_library_path=$gcc_library_path
    remaining=$loaded
    while [[ -n $remaining ]]; do
        entry=${remaining%%:*}
        if [[ :$effective_library_path: != *":$entry:"* ]]; then
            effective_library_path=${effective_library_path:+$effective_library_path:}$entry
        fi
        [[ $remaining == *:* ]] || break
        remaining=${remaining#*:}
    done
    printf 'CPU_GCC_PRIORITY_LIBRARY_PATH=%s\n' "$gcc_library_path"
    export LD_LIBRARY_PATH=$effective_library_path
    printf 'CPU_LD_LIBRARY_PATH_EFFECTIVE=%s\n' "$LD_LIBRARY_PATH"
}

build_profile() (
    profile=$1
    # Each profile starts from the same parent environment. Purge affects this
    # disposable child shell only; no existing owner terminal is changed.
    module purge
    if [[ $profile == cpu ]]; then
        module show gcc/14.2.1
        module show aocl/5.1.0/gcc14.2.1
        before_gcc_library_path=${LD_LIBRARY_PATH:-}
        printf 'CPU_LD_LIBRARY_PATH_BEFORE_GCC=%s\n' "$before_gcc_library_path"
        validate_library_path before-GCC "$before_gcc_library_path"
        module load gcc/14.2.1
        after_gcc_library_path=${LD_LIBRARY_PATH:-}
        printf 'CPU_LD_LIBRARY_PATH_AFTER_GCC=%s\n' "$after_gcc_library_path"
        validate_library_path after-GCC "$after_gcc_library_path"
        module load aocl/5.1.0/gcc14.2.1
        printf 'CPU_LD_LIBRARY_PATH_AFTER_AOCL=%s\n' "${LD_LIBRARY_PATH:-}"
        validate_library_path after-AOCL "${LD_LIBRARY_PATH:-}"
        prioritize_gcc_libraries "$before_gcc_library_path" "$after_gcc_library_path" "${LD_LIBRARY_PATH:-}"
        compiler_c=$(command -v gcc) || fail 'loaded GCC is unavailable'
        compiler_cxx=$(command -v g++) || fail 'loaded G++ is unavailable'
        [[ $("$compiler_c" -dumpfullversion) == 14.2.1 ]] || fail 'loaded GCC version is not 14.2.1'
        [[ $("$compiler_cxx" -dumpfullversion) == 14.2.1 ]] || fail 'loaded G++ version is not 14.2.1'
        extra=(--profile "$profile")
    else
        module show rocm/7.2.0
        module load rocm/7.2.0
        compiler_c=$(command -v amdclang) || fail 'ROCm AMD Clang C is unavailable'
        compiler_cxx=$(command -v amdclang++) || fail 'ROCm AMD Clang C++ is unavailable'
        rocm_root=${ROCM_PATH:-${HIP_PATH:-}}
        [[ $rocm_root == /* && -d $rocm_root ]] || fail 'ROCm module did not provide a usable ROCM_PATH/HIP_PATH'
        # Reuse the target observed in the successful allocated-node build.
        # No login-node GPU query and no xnack/numerical-mode change.
        extra=(--profile "$profile" --rocm-root "$rocm_root" --gpu-arch gfx942:sramecc+:xnack-)
    fi
    [[ $compiler_c == /* && $compiler_cxx == /* ]] || fail 'compiler lookup did not produce absolute paths'
    python=$(command -v python3) || fail 'Python 3 is unavailable'
    printf '\nBUILD_PROFILE=%s\n' "$profile"
    module -t list
    "$compiler_c" --version
    "$compiler_cxx" --version
    "$python" -B "$source_root/jobs/sirius/rocfft_build_only.py" \
        --source-root "$source_root" --run-root "$run_root" --run-id "$run_id" \
        --c-compiler "$compiler_c" --cxx-compiler "$compiler_cxx" \
        "${extra[@]}"
)

build_profile cpu
build_profile hip
build_profile openmp
printf 'GPU_SUITE_SIRIUS_BUILD_ONLY_COMPLETE %s\n' "$run_id"
# Deliberately end here: no examples, benchmarks, GPU query or scheduler call.
