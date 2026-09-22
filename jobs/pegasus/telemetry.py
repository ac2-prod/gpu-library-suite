#!/usr/bin/env python3
"""Parse timestamped GPU telemetry and correlate samples with trial intervals."""

import argparse
import csv
import io
import os
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from gpu_suite.results_io import load_raw_results  # noqa: E402
from gpu_suite.schema import TIMESTAMP_RE  # noqa: E402
from gpu_suite.strict_json import dump_bytes  # noqa: E402
from job_config import CPU_ENVIRONMENT_KEYS  # noqa: E402


class TelemetryError(ValueError):
    pass


OFFSET_RE = re.compile(r"^([+-])(\d{2}):(\d{2})$")
LOCAL_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?$")


def parse_utc_timestamp(value: str) -> datetime:
    if TIMESTAMP_RE.fullmatch(value) is None:
        raise TelemetryError("invalid UTC millisecond timestamp")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def format_utc_timestamp(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    milliseconds = utc.microsecond // 1000
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + "{0:03d}Z".format(milliseconds)


def parse_utc_offset(value: str) -> timezone:
    match = OFFSET_RE.fullmatch(value)
    if match is None:
        raise TelemetryError("invalid UTC offset")
    hours = int(match.group(2))
    minutes = int(match.group(3))
    if hours > 23 or minutes > 59:
        raise TelemetryError("invalid UTC offset")
    sign = 1 if match.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _fields(line: str) -> List[str]:
    text = line.lstrip("#").strip()
    if "," in text:
        return [value.strip() for value in next(csv.reader(io.StringIO(text)))]
    return text.split()


def _parse_local_time(value: str) -> Tuple[time, float]:
    match = LOCAL_TIME_RE.fullmatch(value)
    if match is None:
        raise TelemetryError("invalid local telemetry time: " + value)
    hour, minute, second = (int(match.group(index)) for index in (1, 2, 3))
    if hour > 23 or minute > 59 or second > 59:
        raise TelemetryError("invalid local telemetry time: " + value)
    fraction = (match.group(4) or "").ljust(6, "0")
    microsecond = int(fraction or "0")
    return time(hour, minute, second, microsecond), (
        hour * 3600.0 + minute * 60.0 + second + microsecond / 1000000.0
    )


def parse_dmon_text(
    text: str, start_local_date: str, utc_offset: str,
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    try:
        current_date = date.fromisoformat(start_local_date)
    except ValueError as error:
        raise TelemetryError("invalid start local date") from error
    local_zone = parse_utc_offset(utc_offset)
    header = None  # type: Optional[List[str]]
    samples = []  # type: List[Dict[str, Any]]
    previous_seconds = None  # type: Optional[float]
    rollovers = 0
    for line in text.replace("\r\n", "\n").splitlines():
        if line.strip() == "":
            continue
        fields = _fields(line)
        lower = [field.lower() for field in fields]
        if header is None:
            if "time" in lower:
                if len(fields) != len(set(lower)):
                    raise TelemetryError("duplicate telemetry header")
                header = fields
            continue
        if lower == [field.lower() for field in header]:
            continue
        if line.lstrip().startswith("#"):
            continue
        if len(fields) != len(header):
            raise TelemetryError("telemetry row does not match its header")
        values = dict(zip(header, fields))
        time_name = header[[field.lower() for field in header].index("time")]
        local_time, seconds = _parse_local_time(values[time_name])
        if previous_seconds is not None and seconds < previous_seconds:
            if previous_seconds - seconds < 12.0 * 3600.0:
                raise TelemetryError("telemetry local clock moved backwards")
            current_date += timedelta(days=1)
            rollovers += 1
        previous_seconds = seconds
        local_datetime = datetime.combine(current_date, local_time).replace(
            tzinfo=local_zone
        )
        samples.append({
            "columns": values,
            "sample_index": len(samples),
            "timestamp_utc": format_utc_timestamp(local_datetime),
        })
    if header is None:
        raise TelemetryError("telemetry header with a time column was not found")
    return header, samples, rollovers


def correlate_trials(
    records: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    sample_times = [
        parse_utc_timestamp(str(sample["timestamp_utc"])) for sample in samples
    ]
    correlations = []
    for record in records:
        start_value = record["measurement_start_timestamp"]
        end_value = record["measurement_end_timestamp"]
        if start_value is None or end_value is None:
            indices = []
        else:
            start = parse_utc_timestamp(start_value)
            end = parse_utc_timestamp(end_value)
            if end < start:
                raise TelemetryError("trial measurement interval is reversed")
            indices = [
                index for index, sample_time in enumerate(sample_times)
                if start <= sample_time <= end
            ]
        correlations.append({
            "benchmark": record["benchmark"],
            "hostname": record["hostname"],
            "implementation": record["implementation"],
            "measurement_end_timestamp": end_value,
            "measurement_start_timestamp": start_value,
            "sample_count": len(indices),
            "sample_indices": indices,
            "scope": record["scope"],
            "trial": record["trial"],
        })
    return correlations


def write_telemetry_artifacts(
    raw_path: Path, metadata_output: Path, samples_output: Path,
    correlation_output: Optional[Path], raw_results: Optional[Path],
    telemetry_start: str, telemetry_end: str, sample_interval_sec: float,
    timezone_name: str, utc_offset: str, start_local_date: str,
    command_status: str, capture_mode: str,
    environment: Mapping[str, str] = os.environ,
) -> Dict[str, Any]:
    start = parse_utc_timestamp(telemetry_start)
    end = parse_utc_timestamp(telemetry_end)
    if end < start:
        raise TelemetryError("telemetry end precedes start")
    if sample_interval_sec <= 0.0:
        raise TelemetryError("sample interval must be positive")
    parse_message = None
    header = []  # type: List[str]
    samples = []  # type: List[Dict[str, Any]]
    rollovers = 0
    try:
        header, samples, rollovers = parse_dmon_text(
            raw_path.read_text(encoding="utf-8"), start_local_date, utc_offset
        )
        parse_status = "success"
    except (OSError, ValueError) as error:
        parse_status = "parse-failure"
        parse_message = str(error)
    status = parse_status if command_status == "success" else command_status
    cpu_environment = {
        name: environment.get(name) for name in CPU_ENVIRONMENT_KEYS
    }
    metadata = {
        "command_status": command_status,
        "capture_mode": capture_mode,
        "cpu_runtime_environment": cpu_environment,
        "cpu_thread_semantics": {
            "requested_threads": int(cpu_environment["OMP_NUM_THREADS"])
            if cpu_environment["OMP_NUM_THREADS"] is not None else None,
            "serial_cpu_baseline_effective_threads": 1,
        },
        "midnight_rollover_count": rollovers,
        "parse_message": parse_message,
        "raw_columns": header,
        "sample_interval_sec": sample_interval_sec,
        "start_local_date": start_local_date,
        "telemetry_end_timestamp_utc": telemetry_end,
        "telemetry_metadata_schema_version": 1,
        "telemetry_start_timestamp_utc": telemetry_start,
        "telemetry_status": status,
        "timezone": timezone_name,
        "utc_offset": utc_offset,
    }
    with samples_output.open("xb") as stream:
        stream.write(dump_bytes({
            "samples": samples, "telemetry_samples_schema_version": 1
        }))
    if correlation_output is not None:
        records = [] if raw_results is None or not raw_results.is_file() else load_raw_results(raw_results)
        correlations = correlate_trials(records, samples)
        with correlation_output.open("xb") as stream:
            stream.write(dump_bytes({
                "correlations": correlations,
                "telemetry_correlation_schema_version": 1,
            }))
    with metadata_output.open("xb") as stream:
        stream.write(dump_bytes(metadata))
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--correlation-output", type=Path)
    parser.add_argument("--raw-results", type=Path)
    parser.add_argument("--telemetry-start", required=True)
    parser.add_argument("--telemetry-end", required=True)
    parser.add_argument("--sample-interval-sec", required=True, type=float)
    parser.add_argument("--timezone", required=True)
    parser.add_argument("--utc-offset", required=True)
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument(
        "--command-status", required=True,
        choices=("success", "command-failure", "unavailable"),
    )
    parser.add_argument("--capture-mode", required=True,
                        choices=("csv", "plain-fallback", "unavailable"))
    arguments = parser.parse_args(argv)
    try:
        write_telemetry_artifacts(
            arguments.raw, arguments.metadata_output, arguments.samples_output,
            arguments.correlation_output, arguments.raw_results,
            arguments.telemetry_start, arguments.telemetry_end,
            arguments.sample_interval_sec, arguments.timezone,
            arguments.utc_offset, arguments.start_local_date,
            arguments.command_status, arguments.capture_mode,
        )
    except (OSError, ValueError) as error:
        print("telemetry metadata failed: {0}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
