"""Stage, own, validate and replace a PowerPoint QA export. No Office automation here."""

import argparse
from contextlib import suppress
from datetime import datetime, timezone
import getpass
import json
import os
import sys
from pathlib import Path
import shutil
import tempfile
import platform
import plistlib
from pypdf import PdfReader
from office_package import Package, AuditError, require, file_hash, load_json, local_path, new_json


def lock_path():
    if os.name == "nt":
        return local_path(Path(os.environ["LOCALAPPDATA"]) / "hairdesc-powerpoint-render.lock")
    # Keep one user-wide lock regardless of the selected run/exchange directory.
    legacy = Path("/private/tmp") / ("hairdesc-powerpoint-render-%d.lock" % os.getuid())
    require(not os.path.lexists(legacy), "Previous renderer lock exists; inspect " + str(legacy))
    base = local_path(Path.home() / "Documents" / "HAIRDESC-PPTX-Work")
    base.mkdir(mode=0o700, exist_ok=True)
    require(base.stat().st_uid == os.getuid() and not (base.stat().st_mode & 0o022),
            "PowerPoint lock directory must be owned by the current user and privately writable")
    return base / ".powerpoint-render.lock"


def stamp(out, checksum):
    return Path(str(out) + "." + checksum + ".render.json")


def font_catalog():
    """Host-side identity only; PDF glyph checks determine what PowerPoint actually used."""
    from office_package import digest

    if sys.platform == "darwin":
        import ctypes as c

        cf = c.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        ct = c.CDLL("/System/Library/Frameworks/CoreText.framework/CoreText")
        ct.CTFontManagerCopyAvailablePostScriptNames.restype = c.c_void_p
        cf.CFArrayGetCount.argtypes = [c.c_void_p]
        cf.CFArrayGetCount.restype = c.c_long
        cf.CFArrayGetValueAtIndex.argtypes = [c.c_void_p, c.c_long]
        cf.CFArrayGetValueAtIndex.restype = c.c_void_p
        cf.CFStringGetCString.argtypes = [c.c_void_p, c.c_void_p, c.c_long, c.c_uint32]
        cf.CFStringGetCString.restype = c.c_bool
        cf.CFRelease.argtypes = [c.c_void_p]
        items = ct.CTFontManagerCopyAvailablePostScriptNames()
        names = []
        require(items, "CoreText font catalog unavailable")
        try:
            for i in range(cf.CFArrayGetCount(items)):
                value = cf.CFArrayGetValueAtIndex(items, i)
                buf = c.create_string_buffer(4096)
                require(
                    cf.CFStringGetCString(value, buf, len(buf), 0x08000100),
                    "Unreadable PostScript name",
                )
                names.append(buf.value.decode())
        finally:
            cf.CFRelease(items)
    else:
        import winreg

        names = []
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
            except FileNotFoundError:
                continue
            with key:
                for i in range(winreg.QueryInfoKey(key)[1]):
                    name, value, _ = winreg.EnumValue(key, i)
                    names.append(name + "=" + str(value))
    return digest("\n".join(sorted(names)).encode())


def office_temp():
    base = Path.home() / "Library/Containers/com.microsoft.Powerpoint/Data/tmp"
    require(
        base.is_dir() and not base.is_symlink() and base.stat().st_uid == os.getuid(),
        "Cannot identify the current user's PowerPoint temporary directory",
    )
    return local_path(base)


def prepare(source, output, *, mac_temp=False, mode="normal", exchange=None, owner_pid=None, fixed_exchange=False):
    source = local_path(source)
    out = local_path(output)
    require(
        source.is_file()
        and source.suffix.lower() == ".pptx"
        and out.suffix.lower() == ".pdf"
        and out.parent.is_dir(),
        "Expected a local PPTX and a PDF in an existing local QA directory",
    )
    lock = lock_path()
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        raise AuditError("PowerPoint render lock busy; stop. Inspect " + str(lock))
    job = None
    try:
        previous = None
        if out.exists():
            require(out.is_file(), "Invalid existing PDF")
            previous = file_hash(out)
            metadata = stamp(out, previous)
            require(metadata.is_file(), "Unowned existing PDF/stamp")
            owned = load_json(metadata)
            require(
                owned.get("output") == str(out) and owned.get("pdf_sha256") == previous,
                "Unowned or modified existing PDF",
            )
        directory = local_path(exchange) if exchange else office_temp() if mac_temp else out.parent
        require(
            directory.is_dir()
            and (
                not exchange or (sys.platform == "darwin" and directory.stat().st_uid == os.getuid())
            ),
            "Use an existing private local exchange directory owned by the current user",
        )
        require(
            directory.stat().st_dev == out.parent.stat().st_dev,
            "Temporary PDF and output must share a local filesystem",
        )
        job = Path(tempfile.mkdtemp(prefix="hairdesc-render-", dir=directory))
        srchash = file_hash(source)
        temp = job / (job.name + ".pptx")
        if fixed_exchange:
            require(sys.platform == "darwin" and directory == fixed_directory(),
                    "Fixed exchange must use ~/Documents/HAIRDESC-PPTX-Work/exchange")
            require(source.parent != directory and out.parent != directory,
                    "Source/final output must be outside the reserved exchange directory")
            pkg = Package(source)
        else:
            shutil.copyfile(source, temp)
            require(file_hash(source) == srchash == file_hash(temp), "Source changed during copy")
            pkg = Package(temp)
        info = {
            "job": str(job),
            "source": str(source),
            "source_sha256": srchash,
            "output": str(out),
            "expected_pages": len(pkg.slides),
            "hidden_pages": pkg.hidden,
            "include_hidden": True,
            "previous_pdf": previous,
            "mac_temp": mac_temp,
            "exchange": str(directory) if exchange else None,
            "mode": mode,
            "fixed_exchange": fixed_exchange,
        }
        if fixed_exchange:
            info.update(private_input=str(directory / "render-input.pptx"),
                        temporary_pdf=str(directory / "render-output.pending.pdf"),
                        run_id=job.name)
        lock_record = {"job": str(job)}
        if owner_pid is not None:
            require(type(owner_pid) is int and owner_pid > 0, "Invalid lock owner PID")
            info["owner_pid"] = owner_pid
            lock_record["owner_pid"] = owner_pid
        new_json(job / "owner.json", info)
        new_json(lock / "owner.json", lock_record)
        return job
    except BaseException:
        if job is not None:
            shutil.rmtree(job)
        with suppress(FileNotFoundError):
            (lock / "owner.json").unlink()
        lock.rmdir()
        raise


def owner_info(job):
    job = local_path(job)
    lock = lock_path()
    info = load_json(job / "owner.json")
    expected = {"job": str(job)}
    if "owner_pid" in info:
        expected["owner_pid"] = info["owner_pid"]
    require(
        not lock.is_symlink() and load_json(lock / "owner.json") == expected,
        "Lock ownership mismatch",
    )
    allowed = (
        local_path(info["exchange"])
        if info.get("exchange")
        else office_temp() if info.get("mac_temp") else Path(info["output"]).parent
    )
    require(info["job"] == str(job) and allowed == job.parent, "Stage ownership mismatch")
    return job, lock, info



def fixed_directory():
    return local_path(Path.home() / "Documents/HAIRDESC-PPTX-Work/exchange")


def fixed_state(info):
    directory = fixed_directory()
    require(info.get("fixed_exchange") is True and info.get("exchange") == str(directory)
            and info.get("private_input") == str(directory / "render-input.pptx")
            and info.get("temporary_pdf") == str(directory / "render-output.pending.pdf"),
            "Fixed exchange ownership mismatch")
    return local_path(directory / "render-state.json")


def stage_fixed_input(job):
    """Call only after the lock-held native idle check; do not open Office here."""
    _, _, info = owner_info(job)
    state = fixed_state(info)
    private, pending = map(local_path, (info["private_input"], info["temporary_pdf"]))
    require(not os.path.lexists(pending), "Previous pending PDF exists; inspect before reuse")
    previous = None
    if private.exists():
        require(private.is_file() and private.stat().st_nlink == 1 and state.is_file(),
                "Unowned fixed input; stop")
        previous = load_json(state)
        require(previous.get("status") == "COMPLETE" and previous.get("private_input") == str(private)
                and previous.get("input_sha256") == file_hash(private),
                "Fixed exchange is incomplete or input changed; stop")
    else:
        require(not state.exists(), "Fixed exchange input missing; stop")
    src = Path(info["source"])
    require(file_hash(src) == info["source_sha256"], "Source changed before copy")
    record = dict(status="COPYING", run_id=info["run_id"], private_input=str(private),
                  input_sha256=info["source_sha256"], previous_run_id=(previous or {}).get("run_id"))
    state.write_text(json.dumps(record, indent=2))
    # Reuse the existing input inode; only this lock owner can write it.
    shutil.copyfile(src, private)
    require(file_hash(src) == file_hash(private) == info["source_sha256"], "Source changed during copy")
    record.update(status="READY", input_inode=private.stat().st_ino)
    state.write_text(json.dumps(record, indent=2))
    return private, pending


def complete_fixed_exchange(job):
    """Call only after own presentation close and absence have been confirmed."""
    _, _, info = owner_info(job)
    state = fixed_state(info)
    record = load_json(state)
    require(record.get("run_id") == info["run_id"] and record.get("status") == "READY",
            "Fixed exchange run ownership mismatch")
    require(file_hash(info["private_input"]) == info["source_sha256"]
            and not os.path.lexists(info["temporary_pdf"]), "Fixed exchange not ready for release")
    result = load_json(Path(job) / "result.json")
    record.update(status="COMPLETE", pdf_sha256=result["pdf_sha256"], output=info["output"])
    state.write_text(json.dumps(record, indent=2))


def publish(job, office_version=None):
    job, _, info = owner_info(job)
    pdf = local_path(info["temporary_pdf"]) if info.get("fixed_exchange") else job / (job.name + ".pdf")
    if info.get("fixed_exchange"):
        state = load_json(fixed_state(info))
        require(state.get("status") == "READY" and state.get("run_id") == info["run_id"],
                "Pending PDF run ownership mismatch")
        require(file_hash(info["private_input"]) == info["source_sha256"], "Fixed input changed")
    out = Path(info["output"])
    require(
        pdf.is_file() and not pdf.is_symlink() and pdf.stat().st_size > 0,
        "Empty or missing temporary PDF",
    )
    with pdf.open("rb") as stream:
        require(stream.read(5) == b"%PDF-", "Invalid temporary PDF")
        stream.seek(0)
        reader = PdfReader(stream, strict=True)
        require(
            not reader.is_encrypted
            and len(reader.pages) == info["expected_pages"]
            and len(reader.pages) > 0,
            "PowerPoint PDF page count mismatch",
        )
        for page in reader.pages:
            require(page.mediabox.width > 0 and page.mediabox.height > 0, "Invalid PDF page box")
            contents = page.get_contents()
            if contents is not None:
                contents.get_data()
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(pdf) as doc:
        for page in doc:
            bitmap = page.render(scale=0.25)
            bitmap.close()
            page.close()
    require(file_hash(info["source"]) == info["source_sha256"], "Input changed during rendering")
    if sys.platform == "darwin":
        plist = Path("/Applications/Microsoft PowerPoint.app/Contents/Info.plist")
        with plist.open("rb") as f:
            version = plistlib.load(f)
        office_version = (
            version["CFBundleShortVersionString"] + " (" + version["CFBundleVersion"] + ")"
        )
    require(office_version, "Office version required")
    from office_package import digest

    fingerprint = digest(
        (platform.node() + "|" + platform.platform() + "|" + getpass.getuser()).encode()
    )
    record = dict(
        info,
        pdf_sha256=file_hash(pdf),
        host_fingerprint=fingerprint,
        office_version=office_version,
        font_catalog_sha256=font_catalog(),
        validated_at=datetime.now(timezone.utc).isoformat(),
    )
    new_json(job / "result.json", record)
    # The stamp is committed first and addressed by the PDF's content hash. A crash
    # before switching the PDF leaves the old file and its matching stamp usable.
    new_stamp = stamp(out, record["pdf_sha256"])
    if new_stamp.exists():
        saved = load_json(new_stamp)
        require(
            all(
                saved.get(k) == record[k]
                for k in (
                    "source_sha256",
                    "pdf_sha256",
                    "mode",
                    "host_fingerprint",
                    "office_version",
                    "font_catalog_sha256",
                    "output",
                )
            ),
            "Conflicting ownership record for identical PDF; use a fresh work output",
        )
    else:
        os.link(job / "result.json", new_stamp)
    if out.exists():
        require(
            info["previous_pdf"] is not None and info["previous_pdf"] == file_hash(out),
            "Existing output changed during render",
        )
        os.replace(pdf, out)
    else:
        require(info["previous_pdf"] is None, "Previous PDF disappeared")
        os.link(pdf, out)
        if info.get("fixed_exchange"):
            pdf.unlink()  # never leave the final PDF linked to the reusable pending path
    return out


def finish(job):
    job, lock, info = owner_info(job)
    if not info.get("fixed_exchange"):
        shutil.rmtree(job)
    # Fixed exchange run metadata is retained independently of its reusable files.
    (lock / "owner.json").unlink()
    lock.rmdir()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--mac-temp", action="store_true")
    p.add_argument("--exchange", help="Owner-approved local PowerPoint exchange directory")
    p.add_argument("--mode", choices=("normal", "font-probe", "refit"), default="normal")
    p = sub.add_parser("publish")
    p.add_argument("stage")
    p.add_argument("--office-version")
    p = sub.add_parser("finish")
    p.add_argument("stage")
    a = cli.parse_args()
    if a.command == "prepare":
        print(prepare(a.source, a.output, mac_temp=a.mac_temp, mode=a.mode, exchange=a.exchange))
    elif a.command == "publish":
        print(publish(a.stage, a.office_version))
    else:
        finish(a.stage)
