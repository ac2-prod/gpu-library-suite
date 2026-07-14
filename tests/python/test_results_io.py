import tempfile
import unittest
from pathlib import Path
import csv
import io

from gpu_suite.results_io import ExclusiveRawWriter, parse_stdout_prefix
from gpu_suite.schema import RAW_FIELDS
from gpu_suite.strict_json import dumps

from support import raw_success


class ResultIoTests(unittest.TestCase):
    def test_csv_has_one_header_and_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.csv"
            with ExclusiveRawWriter(path, "csv") as writer:
                writer.write(raw_success(trial=0))
                writer.write(raw_success(trial=1))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split(","), list(RAW_FIELDS))
            self.assertEqual(sum(line.startswith("result_schema_version,") for line in lines), 1)
            with self.assertRaises(FileExistsError):
                with ExclusiveRawWriter(path, "csv"):
                    pass

    def test_stdout_contamination_preserves_valid_prefix(self):
        first = raw_success(trial=0)
        second = raw_success(trial=1)
        stdout = dumps(first) + "\n" + dumps(second) + "\nhuman progress\n"
        records, error = parse_stdout_prefix(stdout, "jsonl")
        self.assertEqual([record["trial"] for record in records], [0, 1])
        self.assertIn("line 3", error)
        self.assertIn("human progress", error)

    def test_duplicate_csv_header_is_contamination_not_a_second_header(self):
        record = raw_success()
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(RAW_FIELDS)
        row = []
        for name in RAW_FIELDS:
            value = record[name]
            if value is None:
                row.append("")
            elif name in {"attempted", "git_dirty"}:
                row.append("true" if value else "false")
            elif name in {
                "implementation_order", "parameters", "verification_metrics",
                "verification_thresholds",
            }:
                row.append(dumps(value))
            else:
                row.append(str(value))
        writer.writerow(row)
        writer.writerow(RAW_FIELDS)
        records, error = parse_stdout_prefix(stream.getvalue(), "csv")
        self.assertEqual(len(records), 1)
        self.assertIn("row 3", error)


if __name__ == "__main__":
    unittest.main()
