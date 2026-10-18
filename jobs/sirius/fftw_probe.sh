#!/usr/bin/env bash
# Read-only inventory, executed on Sirius only after separate owner approval.
# No connection, module change, scheduler operation, build or library execution.
set -uo pipefail

if [[ ${GPU_SUITE_FFTW_PROBE_APPROVED:-} != yes ]]; then
    printf '%s\n' 'FFTW inventory requires separate owner approval.' >&2
    exit 2
fi
for prefix in "$@"; do
    if [[ $prefix != /* ]]; then
        printf '%s\n' 'Each supplied prefix must be an absolute, observed path.' >&2
        exit 2
    fi
done

# Keep the static reader beside this script when preparing a hash-bound payload.
script_dir=${BASH_SOURCE[0]%/*}
[[ $script_dir != "${BASH_SOURCE[0]}" ]] || script_dir=.
module_reader=$script_dir/fftw_module_inventory.py
if [[ ! -f $module_reader ]] || ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'Static module inventory requires the companion fftw_module_inventory.py and Python 3.9+; no queries performed.' >&2
    exit 2
fi

run_query() {
    printf '\nQUERY'
    printf ' %q' "$@"
    printf '\n'
    "$@"
    local status=$?
    printf 'QUERY_EXIT=%s\n' "$status"
    return "$status"
}

filter_provider() {
    local line found=no
    while IFS= read -r line || [[ -n $line ]]; do
        case "$line" in
            *[Ff][Ff][Tt][Ww]*|*[Aa][Oo][Cc][Ll]*) printf '%s\n' "$line"; found=yes ;;
        esac
    done
    [[ $found == yes ]] || printf '%s\n' '(no FFTW/AOCL match; absence is not proof of no installation)'
}

inspect_prefix() {
    local prefix=$1 path
    [[ $prefix == /* ]] || { printf 'IGNORED_NONABSOLUTE_PREFIX=%q\n' "$prefix"; return; }
    printf '\nPREFIX=%q\n' "$prefix"
    for path in "$prefix/include/fftw3.h" "$prefix/lib/pkgconfig/fftw3f.pc" \
                "$prefix/lib64/pkgconfig/fftw3f.pc"; do
        if [[ -f $path ]]; then
            printf 'CANDIDATE_FILE=%q READABLE=%s\n' "$path" "$([[ -r $path ]] && printf yes || printf no)"
        fi
    done
    # Filename inspection only. Do not load, hash or execute candidate libraries.
    for path in "$prefix"/{lib,lib64}/libfftw3f{,_threads,_omp}.{so,so.*,a}; do
        [[ -f $path ]] && printf 'CANDIDATE_LIBRARY=%q\n' "$path"
    done
    return 0
}

printf '%s\n' 'FFTW_READ_ONLY_INVENTORY_BEGIN'
run_query id -un
run_query hostname
run_query date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'HOME=%q\nPATH=%q\n' "${HOME:-}" "${PATH:-}"
for variable in FFTW_ROOT FFTW3f_ROOT FFTW3F_ROOT FFTW3_ROOT CMAKE_PREFIX_PATH \
                CMAKE_INCLUDE_PATH CMAKE_LIBRARY_PATH PKG_CONFIG_PATH \
                PKG_CONFIG_LIBDIR CPATH LIBRARY_PATH LD_LIBRARY_PATH MODULEPATH \
                MODULESHOME LMOD_CMD LOADEDMODULES; do
    printf 'ENV %s=%q\n' "$variable" "${!variable-}"
done

printf '\n%s\n' 'MODULE_INSPECTION=static (no profile, module initialization or modulefile evaluation)'
run_query python3 -I -B "$module_reader"
module_scan_status=$?

if command -v pkg-config >/dev/null 2>&1; then
    printf '\nPKG_CONFIG_COMMAND=%s\n' "$(command -v pkg-config)"
    for option in --modversion --cflags --libs --variable=prefix --variable=includedir --variable=libdir; do
        run_query pkg-config "$option" fftw3f
    done
    pkg_prefix=$(pkg-config --variable=prefix fftw3f 2>/dev/null)
    status=$?
    [[ $status != 0 || -z $pkg_prefix ]] || inspect_prefix "$pkg_prefix"
else
    printf '%s\n' 'PKG_CONFIG_UNAVAILABLE'
fi

if command -v ldconfig >/dev/null 2>&1; then
    printf '\nQUERY ldconfig -p (FFTW lines only)\n'
    cache=$(ldconfig -p 2>&1)
    status=$?
    filter_provider <<< "$cache"
    [[ $status == 0 ]] || printf '%s\n' "$cache" >&2
    printf 'QUERY_EXIT=%s\n' "$status"
else
    printf '%s\n' 'LDCONFIG_UNAVAILABLE'
fi

for variable in FFTW_ROOT FFTW3f_ROOT FFTW3F_ROOT FFTW3_ROOT; do
    [[ -z ${!variable-} ]] || inspect_prefix "${!variable}"
done
# These are standard prefixes, not inferred site-specific installation paths.
inspect_prefix /usr
inspect_prefix /usr/local
for prefix in "$@"; do inspect_prefix "$prefix"; done
printf '\n%s\n' 'FFTW_READ_ONLY_INVENTORY_END (discovery only; no compile/link validation)'
exit "$module_scan_status"
