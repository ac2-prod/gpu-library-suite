#!/usr/bin/env bash
# Lightweight login-node probe. No GPU initialization, build, or scheduler call.
# Arguments are supplied by remote.py, never evaluated as shell source.
set -u
if [ "$#" -ne 2 ]; then
    printf '%s\n' 'expected repository root and completion token' >&2
    exit 64
fi
expected_root=$1
completion_token=$2
finish_probe() {
    local probe_status=$?
    printf 'GPU_SUITE_PROBE_END %s %s\n' "$completion_token" "$probe_status" >&2
}
trap finish_probe EXIT
export GIT_OPTIONAL_LOCKS=0
if ! cd -P -- "$expected_root"; then
    exit 65
fi
physical_root=$PWD
if ! git_root=$(git -c core.fsmonitor=false rev-parse --show-toplevel); then
    exit 66
fi
if [ "$physical_root" != "$expected_root" ] || [ "$git_root" != "$expected_root" ]; then
    printf '%s\n' 'repository root differs from approved root' >&2
    exit 67
fi
printf '\n[identity]\n'
date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
printf 'repository=%s\n' "$physical_root"
printf '\n[git]\n'
git -c core.fsmonitor=false rev-parse HEAD || exit 68
git -c core.fsmonitor=false \
    -c core.untrackedCache=false \
    status --short --branch --untracked-files=all || exit 68
printf '\n[os-cpu]\n'
uname -srm
getconf _NPROCESSORS_ONLN 2>/dev/null || printf '%s\n' 'CPU count unavailable'
printf '\n[modules]\n'
if command -v module >/dev/null 2>&1; then
    module -t list 2>&1
else
    printf '%s\n' 'module command unavailable in this noninteractive shell'
fi
printf '\n[tools]\n'
for tool in cmake python3 cc c++ clang clang++ amdclang amdclang++ hipcc; do
    if tool_path=$(command -v "$tool" 2>/dev/null); then
        printf '%s=%s\n' "$tool" "$tool_path"
        "$tool" --version 2>&1 | sed -n '1,3p'
    else
        printf '%s=unavailable\n' "$tool"
    fi
done
printf '\n[rocm-hints]\n'
printf 'ROCM_PATH=%s\nHIP_PATH=%s\n' "${ROCM_PATH-unset}" "${HIP_PATH-unset}"
# Inspect only named small version/header files under an already configured root.
# Do not guess an installation path or recursively enumerate libraries.
for root in "${ROCM_PATH-}" "${HIP_PATH-}"; do
    case "$root" in
        /*)
            for suffix in .info/version include/hip/hip_version.h; do
                if [ -f "$root/$suffix" ]; then
                    printf 'file=%s/%s\n' "$root" "$suffix"
                    sed -n '1,40p' "$root/$suffix"
                fi
            done
            ;;
    esac
done
printf '\n[not-probed]\n'
printf '%s\n' 'GPU devices, offload, library linking, scheduler and accounting require later approval.'
