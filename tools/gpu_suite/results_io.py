"""Strict raw-result parsing and single-owner exclusive output."""

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple

from .schema import RAW_FIELDS, validate_raw_result
from .strict_json import dumps, loads


INTEGER_FIELDS = {
    "result_schema_version",
    "wave",
    "node_index",
    "problem_size",
    "secondary_size",
    "cpu_threads_requested",
    "cpu_threads_effective",
    "warmup",
    "repeat",
    "trial",
    "getrf_info",
    "getrs_info",
    "device_id",
    "exit_code",
}
FLOAT_FIELDS = {
    "elapsed_total_sec",
    "elapsed_sec",
    "clock_resolution_sec",
}
BOOLEAN_FIELDS = {"attempted", "git_dirty"}
JSON_FIELDS = {
    "implementation_order",
    "parameters",
    "verification_metrics",
    "verification_thresholds",
}
NULLABLE_STRING_FIELDS = {
    "measurement_start_timestamp",
    "measurement_end_timestamp",
    "scheduler",
    "scheduler_job_id",
    "cpu_backend",
    "cpu_backend_role",
    "cpu_parallelism",
    "failure_origin",
    "verification_primary_metric",
    "gpu_name",
    "gpu_uuid",
    "cuda_driver_version",
    "library_version",
    "cuda_runtime_version",
    "git_diff_sha256",
    "source_snapshot_sha256",
}
NULLABLE_INTEGER_FIELDS = {
    "problem_size",
    "secondary_size",
    "cpu_threads_requested",
    "cpu_threads_effective",
    "getrf_info",
    "getrs_info",
    "device_id",
    "exit_code",
}
NULLABLE_FLOAT_FIELDS = {"elapsed_total_sec", "elapsed_sec"}


class ResultIoError(ValueError):
    pass


def _parse_boolean(value: str, name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ResultIoError("invalid boolean in {0}".format(name))


def parse_csv_record(row: Mapping[str, str]) -> Dict[str, Any]:
    if set(row.keys()) != set(RAW_FIELDS):
        raise ResultIoError("CSV fields do not match raw schema")
    result = {}  # type: Dict[str, Any]
    for name in RAW_FIELDS:
        value = row[name]
        if name in NULLABLE_STRING_FIELDS and value == "":
            result[name] = None
        elif name in INTEGER_FIELDS:
            result[name] = None if name in NULLABLE_INTEGER_FIELDS and value == "" else int(value)
        elif name in FLOAT_FIELDS:
            result[name] = None if name in NULLABLE_FLOAT_FIELDS and value == "" else float(value)
        elif name in BOOLEAN_FIELDS:
            result[name] = _parse_boolean(value, name)
        elif name in JSON_FIELDS:
            result[name] = loads(value)
        else:
            result[name] = value
    return validate_raw_result(result)


def parse_jsonl_stdout(stdout: str) -> List[Dict[str, Any]]:
    records = []  # type: List[Dict[str, Any]]
    for line_number, line in enumerate(stdout.splitlines(), 1):
        if line == "":
            raise ResultIoError("blank JSONL line at {0}".format(line_number))
        value = loads(line)
        records.append(validate_raw_result(value))
    return records


def parse_csv_stdout(stdout: str) -> List[Dict[str, Any]]:
    stream = io.StringIO(stdout)
    reader = csv.DictReader(stream)
    if reader.fieldnames != list(RAW_FIELDS):
        raise ResultIoError("CSV header does not match raw schema")
    records = []  # type: List[Dict[str, Any]]
    for row in reader:
        if None in row:
            raise ResultIoError("CSV row has excess columns")
        records.append(parse_csv_record(row))
    return records


def parse_stdout(stdout: str, output_format: str) -> List[Dict[str, Any]]:
    if output_format == "jsonl":
        return parse_jsonl_stdout(stdout)
    if output_format == "csv":
        return parse_csv_stdout(stdout)
    raise ResultIoError("unsupported output format")


def parse_stdout_prefix(
    stdout: str, output_format: str
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return the valid leading records and the first contamination error."""

    records = []  # type: List[Dict[str, Any]]
    if output_format == "jsonl":
        for line_number, line in enumerate(stdout.splitlines(), 1):
            try:
                if line == "":
                    raise ResultIoError("blank JSONL line")
                records.append(validate_raw_result(loads(line)))
            except (TypeError, ValueError) as error:
                rendered = repr(line[:1024])
                return records, "line {0}: {1}; content={2}".format(
                    line_number, error, rendered
                )
        return records, None
    if output_format == "csv":
        stream = io.StringIO(stdout)
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(RAW_FIELDS):
            return records, "CSV header does not match raw schema: {0}".format(
                repr(reader.fieldnames)
            )
        for row_number, row in enumerate(reader, 2):
            try:
                if None in row:
                    raise ResultIoError("CSV row has excess columns")
                records.append(parse_csv_record(row))
            except (TypeError, ValueError) as error:
                return records, "row {0}: {1}; content={2}".format(
                    row_number, error, repr(dict(row))[:1024]
                )
        return records, None
    return records, "unsupported output format"


def _csv_cell(name: str, value: Any) -> str:
    if value is None:
        return ""
    if name in BOOLEAN_FIELDS:
        return "true" if value else "false"
    if name in JSON_FIELDS:
        return dumps(value)
    return str(value)


class ExclusiveRawWriter:
    """Create one raw file exclusively and write its header at most once."""

    def __init__(self, path: Path, output_format: str):
        self.path = path
        self.output_format = output_format
        self.stream = None  # type: Optional[TextIO]
        self.csv_writer = None  # type: Optional[csv.writer]

    def __enter__(self) -> "ExclusiveRawWriter":
        self.stream = self.path.open("x", encoding="utf-8", newline="")
        if self.output_format == "csv":
            self.csv_writer = csv.writer(self.stream, lineterminator="\n")
            self.csv_writer.writerow(RAW_FIELDS)
        elif self.output_format != "jsonl":
            self.stream.close()
            raise ResultIoError("unsupported output format")
        self.stream.flush()
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        if self.stream is None:
            raise ResultIoError("raw writer is not open")
        validated = validate_raw_result(record)
        if self.output_format == "jsonl":
            self.stream.write(dumps(validated) + "\n")
        else:
            assert self.csv_writer is not None
            self.csv_writer.writerow(
                [_csv_cell(name, validated[name]) for name in RAW_FIELDS]
            )
        self.stream.flush()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def load_raw_results(path: Path, output_format: Optional[str] = None) -> List[Dict[str, Any]]:
    selected = output_format
    if selected is None:
        selected = "csv" if path.suffix.lower() == ".csv" else "jsonl"
    text = path.read_text(encoding="utf-8")
    return parse_stdout(text, selected)
