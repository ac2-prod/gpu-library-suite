#!/bin/bash
set -u
umask 022

repository_root=""
run_root=""
config=""
manifest=""
build_metadata=()
run_id=""
wave=""
system_label=""
scheduler=""
scheduler_job_id=""
runtime_environment_sha256=""
scratch_root=""
git_diff_sha256=""
source_snapshot_sha256=""
run_suite_script=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository-root) repository_root="$2"; shift 2 ;;
    --run-root) run_root="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --manifest) manifest="$2"; shift 2 ;;
    --build-metadata) build_metadata+=("$2"); shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --wave) wave="$2"; shift 2 ;;
    --system-label) system_label="$2"; shift 2 ;;
    --scheduler) scheduler="$2"; shift 2 ;;
    --scheduler-job-id) scheduler_job_id="$2"; shift 2 ;;
    --runtime-environment-sha256) runtime_environment_sha256="$2"; shift 2 ;;
    --scratch-root) scratch_root="$2"; shift 2 ;;
    --git-diff-sha256) git_diff_sha256="$2"; shift 2 ;;
    --source-snapshot-sha256) source_snapshot_sha256="$2"; shift 2 ;;
    --run-suite-script) run_suite_script="$2"; shift 2 ;;
    *) echo "unknown run_node.sh option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${repository_root}" || -z "${run_root}" || -z "${config}" ||
      -z "${manifest}" || ${#build_metadata[@]} -eq 0 || -z "${run_id}" ||
      -z "${wave}" || -z "${system_label}" || -z "${scheduler}" ||
      -z "${scheduler_job_id}" || -z "${runtime_environment_sha256}" ||
      -z "${scratch_root}" ]]; then
  echo "run_node.sh is missing a required option" >&2
  exit 2
fi
if [[ -z "${OMPI_COMM_WORLD_RANK:-}" || ! "${OMPI_COMM_WORLD_RANK}" =~ ^[0-9]+$ ]]; then
  echo "OMPI_COMM_WORLD_RANK is required" >&2
  exit 2
fi
if ! scheduler_job_token="$(
  python3 "${repository_root}/jobs/pegasus/scheduler_job_id.py" token \
    --scheduler "${scheduler}" --job-id "${scheduler_job_id}"
)"; then
  exit 2
fi

node_index="${OMPI_COMM_WORLD_RANK}"
hostname_value="$(hostname)"
node_tools="${repository_root}/jobs/pegasus/node_tools.py"
telemetry_tool="${repository_root}/jobs/pegasus/telemetry.py"
if [[ -z "${run_suite_script}" ]]; then
  run_suite_script="${repository_root}/tools/run_suite.py"
fi
scratch="${scratch_root}/gpu-library-suite-${scheduler_job_token}-${run_id}-${wave}-${hostname_value}"
node_directory="${run_root}/waves/${wave}/nodes/${hostname_value}"

mkdir "${scratch}" || exit 90
mkdir "${scratch}/logs" "${scratch}/telemetry" || exit 90
if [[ -e "${node_directory}" ]]; then
  echo "node output already exists: ${node_directory}" >&2
  exit 90
fi

runner_exit_code=125
termination_signal=""
telemetry_pid=""
telemetry_command_status="unavailable"
telemetry_capture_mode="unavailable"
raw_name="raw-results.jsonl"
telemetry_start="$(python3 "${node_tools}" timestamp)"
telemetry_end="${telemetry_start}"
start_local_date="$(date +%Y-%m-%d)"
timezone_name="$(date +%Z)"
offset_compact="$(date +%z)"
utc_offset="${offset_compact:0:3}:${offset_compact:3:2}"

# Invoked indirectly by the EXIT trap below.
# shellcheck disable=SC2317,SC2329
collect_node() {
  process_exit_code=$?
  trap - EXIT INT TERM HUP
  set +e

  telemetry_end="$(python3 "${node_tools}" timestamp)"
  if [[ -n "${telemetry_pid}" ]]; then
    if kill -0 "${telemetry_pid}" 2>/dev/null; then
      kill "${telemetry_pid}" 2>/dev/null
      wait "${telemetry_pid}" 2>/dev/null
    else
      wait "${telemetry_pid}" 2>/dev/null
      telemetry_command_status="command-failure"
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q >"${scratch}/telemetry/nvidia-smi-after.txt" 2>&1
  fi

  python3 "${telemetry_tool}" \
    --raw "${scratch}/telemetry/gpu-dmon-raw.txt" \
    --metadata-output "${scratch}/telemetry/telemetry-metadata.json" \
    --samples-output "${scratch}/telemetry/gpu-samples.json" \
    --correlation-output "${scratch}/telemetry/trial-correlation.json" \
    --raw-results "${scratch}/${raw_name}" \
    --telemetry-start "${telemetry_start}" --telemetry-end "${telemetry_end}" \
    --sample-interval-sec 1 --timezone "${timezone_name}" \
    --utc-offset "${utc_offset}" --start-local-date "${start_local_date}" \
    --command-status "${telemetry_command_status}" \
    --capture-mode "${telemetry_capture_mode}" \
    >"${scratch}/logs/telemetry-parser.stdout" \
    2>"${scratch}/logs/telemetry-parser.stderr"
  telemetry_tool_exit=$?

  classification_path="${scratch}/raw-classification.json"
  classification_argument=()
  if [[ -f "${scratch}/${raw_name}" ]]; then
    if python3 "${node_tools}" classify --raw "${scratch}/${raw_name}" \
      --node-metadata "${scratch}/node-metadata.json" \
      --output "${classification_path}" \
      >"${scratch}/logs/classification.stdout" \
      2>"${scratch}/logs/classification.stderr"; then
      classification_argument=(--classification "${classification_path}")
    fi
  fi

  tool_status="success"
  if [[ ${telemetry_tool_exit} -ne 0 ]]; then
    tool_status="failure"
  elif [[ "${telemetry_command_status}" != "success" ]]; then
    tool_status="warning"
  fi
  message_arguments=()
  if [[ ${runner_exit_code} -ne 0 ]]; then
    message_arguments+=(--message "suite runner exited with ${runner_exit_code}")
  fi
  if [[ -n "${termination_signal}" ]]; then
    message_arguments+=(--message "node runner received ${termination_signal}")
  fi
  if [[ ${telemetry_tool_exit} -ne 0 ]]; then
    message_arguments+=(--message "telemetry parser failed")
  fi

  if ! mkdir "${node_directory}"; then
    echo "could not create shared node directory" >&2
    exit 90
  fi
  cp -R "${scratch}/." "${node_directory}/"
  copy_exit=$?
  collection_status="success"
  raw_collection_status="missing"
  log_collection_status="failure"
  if [[ ${copy_exit} -ne 0 ]]; then
    collection_status="failure"
  fi
  if [[ -f "${node_directory}/${raw_name}" ]]; then
    raw_collection_status="success"
  else
    collection_status="failure"
  fi
  if [[ -d "${node_directory}/logs" ]]; then
    log_collection_status="success"
  else
    collection_status="failure"
  fi

  status_arguments=(
    status --run-id "${run_id}" --wave "${wave}"
    --node-index "${node_index}" --hostname "${hostname_value}"
    --scheduler "${scheduler}" --scheduler-job-id "${scheduler_job_id}"
    --runner-exit-code "${runner_exit_code}"
    --process-exit-code "${process_exit_code}"
    --collection-status "${collection_status}"
    --raw-collection-status "${raw_collection_status}"
    --log-collection-status "${log_collection_status}"
    --tool-status "${tool_status}"
    --telemetry-metadata "${scratch}/telemetry/telemetry-metadata.json"
  )
  status_arguments+=("${classification_argument[@]}")
  if [[ -n "${termination_signal}" ]]; then
    status_arguments+=(--termination-signal "${termination_signal}")
  fi
  status_arguments+=("${message_arguments[@]}")
  status_arguments+=(--output "${node_directory}/node-status.json")
  python3 "${node_tools}" "${status_arguments[@]}"
  status_exit=$?
  if [[ ${status_exit} -ne 0 || "${collection_status}" != "success" ]]; then
    exit 91
  fi
  exit 0
}

# Invoked indirectly by signal traps below.
# shellcheck disable=SC2317,SC2329
handle_signal() {
  termination_signal="$1"
  exit "$2"
}

trap collect_node EXIT
trap 'handle_signal HUP 129' HUP
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

export GPU_SUITE_NODE_GPU_QUERY_STATUS="unavailable"
export GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC="nvidia-smi is unavailable"
if command -v nvidia-smi >/dev/null 2>&1; then
  node_gpu_name="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader,nounits \
    2>"${scratch}/logs/node-gpu-query.stderr")"
  node_gpu_uuid="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader,nounits \
    2>>"${scratch}/logs/node-gpu-query.stderr")"
  node_driver_version="$(nvidia-smi -i 0 --query-gpu=driver_version \
    --format=csv,noheader,nounits 2>>"${scratch}/logs/node-gpu-query.stderr")"
  if [[ -n "${node_gpu_name}" && -n "${node_gpu_uuid}" &&
        -n "${node_driver_version}" ]]; then
    export GPU_SUITE_GPU_NAME="${node_gpu_name}"
    export GPU_SUITE_GPU_UUID="${node_gpu_uuid}"
    export GPU_SUITE_NVIDIA_DRIVER_VERSION="${node_driver_version}"
    export GPU_SUITE_NODE_GPU_QUERY_STATUS="success"
    export GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC=""
  else
    export GPU_SUITE_NODE_GPU_QUERY_STATUS="failure"
    export GPU_SUITE_NODE_GPU_QUERY_DIAGNOSTIC="device-0 query returned incomplete data"
  fi
fi

if ! python3 "${node_tools}" metadata --config "${config}" --manifest "${manifest}" \
  --run-id "${run_id}" --wave "${wave}" --node-index "${node_index}" \
  --hostname "${hostname_value}" \
  --scheduler "${scheduler}" --scheduler-job-id "${scheduler_job_id}" \
  --runtime-environment-sha256 "${runtime_environment_sha256}" \
  --output "${scratch}/node-metadata.json" \
  >"${scratch}/logs/node-metadata.stdout" \
  2>"${scratch}/logs/node-metadata.stderr"; then
  runner_exit_code=125
  exit 125
fi
raw_name="$(python3 "${node_tools}" raw-name --config "${config}")"

{
  hostname
  date
  command -v lscpu >/dev/null 2>&1 && lscpu
  [[ -r /proc/cpuinfo ]] && cat /proc/cpuinfo
  command -v numactl >/dev/null 2>&1 && numactl --hardware
  command -v taskset >/dev/null 2>&1 && taskset -cp $$
  module list
  command -v nvc >/dev/null 2>&1 && nvc --version
  command -v nvc++ >/dev/null 2>&1 && nvc++ --version
  env | while IFS= read -r line; do
    case "${line}" in
      OMP_NUM_THREADS=*|OMP_PROC_BIND=*|OMP_PLACES=*|OMP_DYNAMIC=*|MKL_NUM_THREADS=*|MKL_DYNAMIC=*|MKL_THREADING_LAYER=*|OPENBLAS_NUM_THREADS=*)
        printf '%s\n' "${line}" ;;
    esac
  done
  if command -v turbostat >/dev/null 2>&1; then
    turbostat --Summary --quiet --interval 1 --num_iterations 1
  elif [[ -d /sys/devices/system/cpu/cpufreq ]]; then
    find /sys/devices/system/cpu/cpufreq -maxdepth 2 -type f -readable -print
  fi
} >"${scratch}/telemetry/cpu-telemetry.txt" 2>&1

: >"${scratch}/telemetry/gpu-dmon-raw.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"${scratch}/telemetry/nvidia-smi-before.txt" 2>&1
  dmon_help="$(nvidia-smi dmon --help 2>&1)"
  if [[ "${dmon_help}" == *"--format"* ]]; then
    telemetry_capture_mode="csv"
    nvidia-smi dmon -s pucv -d 1 -o T --format csv,nounit \
      >"${scratch}/telemetry/gpu-dmon-raw.txt" \
      2>"${scratch}/logs/gpu-dmon.stderr" &
  else
    telemetry_capture_mode="plain-fallback"
    nvidia-smi dmon -s pucv -d 1 -o T \
      >"${scratch}/telemetry/gpu-dmon-raw.txt" \
      2>"${scratch}/logs/gpu-dmon.stderr" &
  fi
  telemetry_pid=$!
  telemetry_command_status="success"
fi

runner_arguments=(
  --config "${config}" --manifest "${manifest}"
  --output "${scratch}/${raw_name}" --run-id "${run_id}"
  --system-label "${system_label}" --wave "${wave}"
  --node-index "${node_index}" --hostname "${hostname_value}"
  --runtime-environment-sha256 "${runtime_environment_sha256}"
  --scheduler "${scheduler}" --scheduler-job-id "${scheduler_job_id}"
)
for metadata_path in "${build_metadata[@]}"; do
  runner_arguments+=(--build-metadata "${metadata_path}")
done
if [[ -n "${git_diff_sha256}" ]]; then
  runner_arguments+=(--git-diff-sha256 "${git_diff_sha256}")
fi
if [[ -n "${source_snapshot_sha256}" ]]; then
  runner_arguments+=(--source-snapshot-sha256 "${source_snapshot_sha256}")
fi

python3 "${run_suite_script}" "${runner_arguments[@]}" \
  >"${scratch}/logs/run-suite.stdout" \
  2>"${scratch}/logs/run-suite.stderr"
runner_exit_code=$?
exit "${runner_exit_code}"
