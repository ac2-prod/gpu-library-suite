#!/bin/bash
set -euo pipefail
umask 022

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
config=""
source_root=""
build_root=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --source) source_root="$2"; shift 2 ;;
    --build) build_root="$2"; shift 2 ;;
    *) echo "unknown build_cpu_cuda.sh option: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "${config}" || -z "${source_root}" || -z "${build_root}" ]]; then
  echo "usage: build_cpu_cuda.sh --config FILE --source DIR --build DIR" >&2
  exit 2
fi
if [[ ! -d "${source_root}" || "${source_root}" != /* || "${build_root}" != /* ]]; then
  echo "source and build paths must be absolute; source must exist" >&2
  exit 1
fi

helper=(python3 "${script_directory}/job_config.py" --config "${config}")
module_purge="$("${helper[@]}" get module_purge)"
if [[ "${module_purge}" == "true" ]]; then
  module purge
fi
mapfile -t build_modules < <("${helper[@]}" modules --profile cpu-cuda)
for module_name in "${build_modules[@]}"; do
  module load "${module_name}"
done

generator="$("${helper[@]}" get cmake_generator)"
build_type="$("${helper[@]}" get build_type)"
parallelism="$("${helper[@]}" get build_parallelism)"
architectures="$("${helper[@]}" get cuda_architectures)"
toolkit_root="$("${helper[@]}" get cuda_toolkit_root)"

mkdir -p "${build_root}"
profile_marker="${build_root}/.gpu-suite-build-profile"
if [[ -e "${profile_marker}" ]]; then
  [[ "$(<"${profile_marker}")" == "cpu-cuda" ]] || {
    echo "build tree belongs to a different profile" >&2
    exit 1
  }
else
  (set -C; printf '%s\n' cpu-cuda >"${profile_marker}")
fi
module list >"${build_root}/module-list.txt" 2>&1

cmake -S "${source_root}" -B "${build_root}" -G "${generator}" \
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_CUDA_ARCHITECTURES="${architectures}" \
  -DCUDAToolkit_ROOT="${toolkit_root}" \
  -DGPU_SUITE_BUILD_CPU=ON \
  -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build "${build_root}" --parallel "${parallelism}" \
  --target gpu_suite_partial_manifest
printf 'CPU/CUDA partial manifest: %s\n' "${build_root}/partial-manifest.json"
