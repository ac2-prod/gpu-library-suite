#!/usr/bin/env bash
# Source only in the approved allocated-job shell; prepares environment only.
# Does not start the job body, query a GPU, build, submit, or connect remotely.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    printf 'Sirius compute modules: source this helper in the allocated job shell\n' >&2
    exit 2
fi

_sirius_compute_module_error() {
    printf 'Sirius compute modules: %s\n' "$*" >&2
    return 2
}

_sirius_compute_validate_libraries() {
    local label=$1 remaining=$2 entry
    [[ -n $remaining ]] || return 0
    while :; do
        entry=${remaining%%:*}
        [[ $entry == /* ]] || { _sirius_compute_module_error "$label contains an empty or relative library directory"; return 2; }
        [[ -d $entry ]] || { _sirius_compute_module_error "$label contains a missing library directory"; return 2; }
        [[ $remaining == *:* ]] || break
        remaining=${remaining#*:}
    done
}

_sirius_compute_prepare_modules() {
    local source_root=${STAGE31_SOURCE_ROOT:-} run_root=${STAGE31_RUN_ROOT:-} run_id=${STAGE31_RUN_ID:-}
    local source_real run_real before_gcc after_gcc after_aocl after_rocm gcc_paths='' effective=''
    local remaining entry cpu_cc cpu_cxx amd_cxx version
    local aocl_root=/system/apps/rhel/9.6-202602/aocl/5.1.0/gcc
    local rocm_root=/opt/rocm-7.2.0
    [[ ${GPU_SUITE_SIRIUS_COMPUTE_APPROVED:-} == yes && -n ${PBS_JOBID:-} ]] || {
        _sirius_compute_module_error 'explicit compute approval and PBS_JOBID are required'; return 2;
    }
    [[ $run_id =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,100}$ && $run_id != *..* ]] || {
        _sirius_compute_module_error 'invalid STAGE31_RUN_ID'; return 2;
    }
    [[ $source_root == /* && -d $source_root && ! -L $source_root && ! -L $source_root/runs &&
       $run_root == "$source_root/runs/$run_id" && -d $run_root && ! -L $run_root ]] || {
        _sirius_compute_module_error 'invalid STAGE31 source/run directories'; return 2;
    }
    source_real=$(cd -- "$source_root" && pwd -P) || return 2
    run_real=$(cd -- "$run_root" && pwd -P) || return 2
    [[ $source_real == "$source_root" && $run_real == "$run_root" ]] || {
        _sirius_compute_module_error 'noncanonical STAGE31 source/run directories'; return 2;
    }
    command -v module >/dev/null 2>&1 || { _sirius_compute_module_error 'module unavailable'; return 2; }
    export PAGER=cat MODULES_PAGER=cat LMOD_PAGER=cat PYTHONDONTWRITEBYTECODE=1
    printf 'COMPUTE_MODULE_ENVIRONMENT USER=%s HOSTNAME=%s HOME=%s PATH=%s\n' \
        "$(id -un)" "$(hostname)" "${HOME:-unavailable}" "${PATH:-unavailable}"
    printf 'COMPUTE_INITIAL_MODULES\n'
    module -t list || return 2
    # The dispatcher invokes this in a disposable allocated-job shell.
    module purge || return 2
    module show gcc/14.2.1 || return 2
    module show aocl/5.1.0/gcc14.2.1 || return 2
    module show rocm/7.2.0 || return 2
    before_gcc=${LD_LIBRARY_PATH:-}
    printf 'COMPUTE_LD_LIBRARY_PATH_BEFORE_GCC=%s\n' "$before_gcc"
    _sirius_compute_validate_libraries before-GCC "$before_gcc" || return 2
    module load gcc/14.2.1 || return 2
    after_gcc=${LD_LIBRARY_PATH:-}
    printf 'COMPUTE_LD_LIBRARY_PATH_AFTER_GCC=%s\n' "$after_gcc"
    _sirius_compute_validate_libraries after-GCC "$after_gcc" || return 2
    remaining=$after_gcc
    while [[ -n $remaining ]]; do
        entry=${remaining%%:*}
        if [[ :$before_gcc: != *":$entry:"* && :$gcc_paths: != *":$entry:"* ]]; then
            gcc_paths=${gcc_paths:+$gcc_paths:}$entry
        fi
        [[ $remaining == *:* ]] || break
        remaining=${remaining#*:}
    done
    [[ -n $gcc_paths ]] || { _sirius_compute_module_error 'GCC module introduced no usable library directories'; return 2; }
    cpu_cc=$(command -v gcc) || { _sirius_compute_module_error 'loaded GCC unavailable'; return 2; }
    cpu_cxx=$(command -v g++) || { _sirius_compute_module_error 'loaded G++ unavailable'; return 2; }
    for entry in "$cpu_cc" "$cpu_cxx"; do
        [[ $entry == /* && -f $entry && -x $entry ]] || { _sirius_compute_module_error 'GCC driver is not an absolute executable'; return 2; }
        version=$("$entry" -dumpfullversion) || return 2
        [[ $version == 14.2.1 ]] || { _sirius_compute_module_error 'GCC driver version is not 14.2.1'; return 2; }
    done
    module load aocl/5.1.0/gcc14.2.1 || return 2
    after_aocl=${LD_LIBRARY_PATH:-}
    printf 'COMPUTE_LD_LIBRARY_PATH_AFTER_AOCL=%s\n' "$after_aocl"
    _sirius_compute_validate_libraries after-AOCL "$after_aocl" || return 2
    [[ ${AOCL_ROOT:-} == "$aocl_root" && -f $aocl_root/include/fftw3.h ]] || {
        _sirius_compute_module_error 'AOCL prefix/header differs from the observed provider'; return 2;
    }
    module load rocm/7.2.0 || return 2
    after_rocm=${LD_LIBRARY_PATH:-}
    printf 'COMPUTE_LD_LIBRARY_PATH_AFTER_ROCM=%s\n' "$after_rocm"
    _sirius_compute_validate_libraries after-ROCm "$after_rocm" || return 2
    [[ ${AOCL_ROOT:-} == "$aocl_root" && ${ROCM_PATH:-} == "$rocm_root" && ${HIP_PATH:-} == "$rocm_root" &&
       -f $rocm_root/include/hip/hip_runtime_api.h && -f $rocm_root/include/rocfft/rocfft.h ]] || {
        _sirius_compute_module_error 'ROCm/AOCL prefixes or headers differ from observed providers'; return 2;
    }
    # Reapply GCC precedence only after ALL module loads. Retain every observed
    # AOCL/ROCm/initial entry even if a later module replaced the whole variable.
    effective=$gcc_paths
    for remaining in "$after_rocm" "$after_aocl" "$after_gcc" "$before_gcc"; do
        while [[ -n $remaining ]]; do
            entry=${remaining%%:*}
            if [[ :$effective: != *":$entry:"* ]]; then
                effective=${effective:+$effective:}$entry
            fi
            [[ $remaining == *:* ]] || break
            remaining=${remaining#*:}
        done
    done
    export LD_LIBRARY_PATH=$effective CPU_CC=$cpu_cc CPU_CXX=$cpu_cxx
    export FFTW3f_ROOT=$aocl_root FFTW3f_THREADING=OPENMP
    amd_cxx=$(command -v amdclang++) || { _sirius_compute_module_error 'loaded AMD Clang unavailable'; return 2; }
    [[ $amd_cxx == "$rocm_root/"* && -f $amd_cxx && -x $amd_cxx ]] || {
        _sirius_compute_module_error 'AMD Clang driver is not in the observed ROCm prefix'; return 2;
    }
    version=$("$amd_cxx" --version) || return 2
    [[ $version == *'AMD clang'* ]] || { _sirius_compute_module_error 'loaded compiler is not AMD Clang'; return 2; }
    export HIP_CLANG=$amd_cxx OPENMP_CXX=$amd_cxx
    printf 'COMPUTE_GCC_PRIORITY_LIBRARY_PATH=%s\n' "$gcc_paths"
    printf 'COMPUTE_LD_LIBRARY_PATH_EFFECTIVE=%s\n' "$LD_LIBRARY_PATH"
    printf 'COMPUTE_FINAL_MODULES\n'
    module -t list || return 2
    "$CPU_CC" --version || return 2
    "$CPU_CXX" --version || return 2
    printf '%s\n' "$version"
    printf 'COMPUTE_FINAL_ENVIRONMENT HOME=%s PATH=%s CPU_CC=%s CPU_CXX=%s FFTW3f_ROOT=%s FFTW3f_THREADING=%s ROCM_PATH=%s HIP_PATH=%s\n' \
        "${HOME:-unavailable}" "$PATH" "$CPU_CC" "$CPU_CXX" "$FFTW3f_ROOT" "$FFTW3f_THREADING" "$ROCM_PATH" "$HIP_PATH"
}

if _sirius_compute_prepare_modules; then
    unset -f _sirius_compute_prepare_modules _sirius_compute_validate_libraries _sirius_compute_module_error
    return 0
else
    unset -f _sirius_compute_prepare_modules _sirius_compute_validate_libraries _sirius_compute_module_error
    return 2
fi
