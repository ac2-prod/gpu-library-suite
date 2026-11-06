"""PowerPoint PDF rendering: one LaunchServices open, bounded read-only readiness queries."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import render_guard as guard
from office_package import file_hash, local_path, require


def powerpoint_process():
    r = subprocess.run(["/bin/ps", "-axo", "pid,lstart,comm"], capture_output=True, text=True, check=True)
    rows = [line.strip() for line in r.stdout.splitlines()
            if line.rstrip().endswith("/Microsoft PowerPoint.app/Contents/MacOS/Microsoft PowerPoint")]
    require(len(rows) <= 1, "Multiple PowerPoint processes; stop")
    return rows[0] if rows else None


def run(source, output, exchange, report_path, font_probe=False, observation=None):
    report_path = local_path(report_path)
    require(not report_path.exists(), "Use a new report path")
    report = dict(status="STARTING", open_requested=False, open_calls=0,
                  presentation_identified=False, pdf_export_started=False,
                  own_presentation_closed=False, lock_released=False, commands=[])
    events = report_path.with_suffix(".events.jsonl")
    require(not events.exists(), "Use a new event log path")
    job = None
    def record(phase):
        report["phase"] = phase
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        with events.open("a") as stream:
            stream.write(json.dumps(dict(utc=report["updated_utc"], phase=phase,
                run_id=report.get("run_id"), open_requested=report["open_requested"],
                presentation_identified=report["presentation_identified"],
                pdf_export_started=report["pdf_export_started"],
                pdf_export_completed=report.get("pdf_export_completed", False),
                own_presentation_closed=report["own_presentation_closed"],
                lock_released=report["lock_released"])) + "\n")
        if job and job.is_dir():
            (job / "execution.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    def command(argv, timeout):
        item = dict(argv=list(map(str, argv)), timeout=timeout)
        report["commands"].append(item); record(report["phase"])
        try:
            r = subprocess.run(item["argv"], capture_output=True, text=True, timeout=timeout)
            item.update(exit_code=r.returncode, stdout=r.stdout, stderr=r.stderr)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            item.update(exit_code=None, terminated_helper=True, stdout=(e.stdout or b"").decode() if isinstance(e.stdout,bytes) else e.stdout or "",
                        stderr=(e.stderr or b"").decode() if isinstance(e.stderr,bytes) else e.stderr or "", error="helper wall timeout")
            return None, item["stdout"], item["stderr"]
        finally:
            record(report["phase"])
    try:
        require(sys.platform == "darwin", "macOS required")
        require(not font_probe, "Font/content changes are disabled for this unmodified-deck test")
        source, output, exchange = map(local_path, (source, output, exchange))
        require(exchange == guard.fixed_directory(), "Use the fixed Documents exchange directory")
        require(report_path.parent != exchange and report_path not in (source, output),
                "Keep reports outside the exchange and input/output files")
        report.update(source=str(source), output=str(output), source_sha256=file_hash(source))
        report["PowerPoint_before"] = powerpoint_process()
        job = guard.prepare(source, output, exchange=exchange,
                            mode="normal", owner_pid=os.getpid(), fixed_exchange=True)
        report.update(job=str(job), run_id=job.name)
        private = exchange / "render-input.pptx"
        pdf = exchange / "render-output.pending.pdf"
        compiled = job / "render.scpt"
        record("compile")
        code, out, err = command(["/usr/bin/osacompile", "-o", compiled, HERE / "render.applescript"], 10)
        require(code == 0, "AppleScript compile failed: " + err)
        require(powerpoint_process() == report["PowerPoint_before"], "PowerPoint process changed during preparation")
        if report["PowerPoint_before"]:
            record("existing-idle-instance-check")
            code, out, err = command(["/usr/bin/osascript", compiled, "idle", private, pdf,
                                      private.name, pdf.name, "no", "no"], 5)
            require(code == 0 and out.strip() == "IDLE", "Existing PowerPoint is not safely idle: " + err)
        record("copy-fixed-input")
        private, pdf = guard.stage_fixed_input(job)
        report.update(private_source_sha256=file_hash(private), fixed_input=str(private),
                      fixed_pdf=str(pdf), fixed_input_inode=private.stat().st_ino,
                      pending_absent_before_open=not pdf.exists())
        # Persist intent BEFORE dispatch. Never resend an uncertain or failed open.
        deadline = time.monotonic() + 120
        report.update(open_requested=True, open_calls=1)
        record("open-request")
        code, out, err = command(["/usr/bin/open", "-a", "/Applications/Microsoft PowerPoint.app", private], 5)
        require(code == 0, "LaunchServices open not confirmed: " + err)
        process = report["PowerPoint_before"]
        def native(action, seconds, allow_pdf=False):
            require(not (report_path.parent / (report_path.stem + ".stop")).exists(), "Owner requested stop; no further AppleEvents")
            if observation and Path(observation).exists():
                require(not json.loads(Path(observation).read_text()).get("repair"), "Repair reported; no further AppleEvents")
            current = powerpoint_process()
            require(current and current == report.get("PowerPoint_process"), "PowerPoint process changed/disappeared; stop")
            return command(["/usr/bin/osascript", compiled, action, private, pdf,
                            private.name, pdf.name, "yes" if allow_pdf else "no",
                            "yes" if font_probe else "no"], seconds)
        record("readiness-poll")
        while time.monotonic() < deadline:
            if (report_path.parent / (report_path.stem + ".stop")).exists():
                raise RuntimeError("Owner requested stop; no further AppleEvents")
            observed = powerpoint_process()
            if process is None and observed:
                process = observed
            require(not observed or observed == process, "PowerPoint identity changed; stop")
            report["PowerPoint_process"] = process
            if observed:
                remaining = deadline - time.monotonic()
                if remaining < 5:
                    time.sleep(max(0, remaining))
                    break
                code, out, err = native("probe", 5)
                if code == 0 and out.startswith("FOUND\t"):
                    report.update(presentation_identified=True, presentation_identity=out.strip())
                    record("presentation-identified")
                    break
                require(not any(x in err for x in ("-1743", "-1744", "-10004", "-128", "AMBIGUOUS_PRIVATE_PRESENTATION")), "Permission denied or ambiguous presentation: " + err)
            time.sleep(min(1, max(0, deadline-time.monotonic())))
        require(report["presentation_identified"], "Presentation readiness deadline exceeded (120s); open was not resent")
        if (report_path.parent / (report_path.stem + ".stop")).exists():
            raise RuntimeError("Owner requested stop; no further AppleEvents")
        if observation:
            record("wait-owner-dialog-observation")
            until = time.monotonic() + 120
            while not Path(observation).exists() and time.monotonic() < until:
                time.sleep(0.5)
            require(Path(observation).exists(), "Owner dialog observation unavailable; no export")
            seen = json.loads(Path(observation).read_text())
            require(seen.get("owner_confirmed") is True and seen.get("repair") is False,
                    "Repair reported or not excluded; no export")
            report["dialog_observation"] = seen
        report["pdf_export_started"] = True; record("export")
        code, out, err = native("export", 65)
        require(code == 0 and out.startswith("EXPORTED\t"), "Export not confirmed: " + err)
        report["pdf_export_completed"] = True
        record("pdf-export-completed")
        record("validate-and-publish")
        report["temporary_pdf_inode"] = pdf.stat().st_ino if pdf.exists() else None
        report["prior_output_inode"] = output.stat().st_ino if output.exists() else None
        guard.publish(job)
        report.update(pdf_sha256=file_hash(output), final_pdf_inode=output.stat().st_ino,
                      pdf_bytes=output.stat().st_size, pdf_published=True)
        require(report["temporary_pdf_inode"] == report["final_pdf_inode"], "PDF publish inode mismatch")
        report["render_record"] = json.loads(guard.stamp(output, file_hash(output)).read_text())
        record("close-owned-presentation")
        code, out, err = native("close", 10, allow_pdf=True)
        require(code == 0 and out.startswith("CLOSED\t"), "Close not confirmed: " + err)
        code, out, err = native("probe", 5, allow_pdf=True)
        require(code == 0 and out.strip() == "ABSENT", "Private presentation disappearance not confirmed")
        report["own_presentation_closed"] = True
        record("own-presentation-closed")
        require(file_hash(source) == report["source_sha256"], "Input PPTX changed")
        require(file_hash(private) == report["source_sha256"], "Private PPTX bytes changed")
        report["private_source_sha256_after"] = file_hash(private)
        guard.complete_fixed_exchange(job)
        record("release-lock")
        guard.finish(job)
        report.update(status="PASS", lock_released=True, PowerPoint_after=powerpoint_process())
        record("complete")
        print(str(output))
        return 0
    except Exception as e:
        report.update(status="STOPPED", error=str(e))
        if job is not None and not report["open_requested"]:
            guard.finish(job); report["lock_released"] = True
        record("stopped")
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source"); p.add_argument("output")
    p.add_argument("--exchange", required=True); p.add_argument("--report", required=True)
    p.add_argument("--font-probe", action="store_true")
    p.add_argument("--observation", help="Owner-supplied dialog observation JSON before export")
    a = p.parse_args()
    sys.exit(run(a.source, a.output, a.exchange, a.report, a.font_probe, a.observation))
