"""Reproduce the translation skill checks using only generated synthetic slides."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from office_package import (
    AuditError,
    ET,
    Package,
    clone_zip,
    file_hash,
    first,
    local_path,
    new_json,
    parse_xml,
    require,
)
from pptx_audit import inventory, template, compare, compare_sdk, sdk_report
from apply_translation import apply
from fixture_deck import create_fixture, EN
from boxed_copy import boxed_copy
from pdf_audit import audit_pdf
import render_guard

HERE = Path(__file__).resolve().parent


def command(args, **kw):
    proc = subprocess.run([str(v) for v in args], capture_output=True, text=True, **kw)
    if proc.returncode:
        raise AuditError(
            f"{args[0]} exited {proc.returncode}: {proc.stderr[-2500:]} {proc.stdout[-800:]}"
        )
    return proc.stdout.strip()


def validator(dotnet, work):
    where = work / "validator"
    shutil.copytree(HERE / "openxml-validator", where)
    feed = work / "offline-feed"
    feed.mkdir()
    env = {
        **os.environ,
        "DOTNET_CLI_HOME": str(work / "dotnet-home"),
        "DOTNET_ROOT": str(Path(dotnet).resolve().parent),
        "DOTNET_NOLOGO": "1",
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }
    cache = Path(os.environ.get("NUGET_PACKAGES", Path(dotnet).resolve().parent.parent / "packages"))
    env["NUGET_PACKAGES"] = str(cache)
    command(
        [dotnet, "restore", "--locked-mode", "--source", feed, "-p:NuGetAudit=false"],
        cwd=where,
        env=env,
    )
    command([dotnet, "build", "-c", "Release", "--no-restore"], cwd=where, env=env)
    return where / "bin/Release/net10.0/Validator.dll"


def bad_copy(source, dest, part, func):
    pkg = Package(source)
    tree = parse_xml(pkg.data[part])
    func(tree)
    clone_zip(source, dest, {part: ET.tostring(tree, encoding="utf-8")})


def rejection(result, name, callback, message):
    try:
        callback()
    except (AuditError, ValueError) as e:
        require(message.lower() in str(e).lower(), name + " failed for the wrong reason: " + str(e))
        result["negative_checks"][name] = str(e)[:450]
        return
    raise AuditError("Expected rejection: " + name)


def run(args, work, result):
    src = work / "ja/render_check.pptx"
    dst = work / "en/render_check.pptx"
    box = work / "boxed/render_check.pptx"
    for p in (src, dst, box):
        p.parent.mkdir()
    create_fixture(src)
    pkg = Package(src)
    sourcehash = pkg.sha
    inv, allow = inventory(pkg, [1, 2])
    require(not inv["blocking"], str(inv["blocking"]))
    new_json(work / "inventory.json", inv)
    new_json(work / "allowlist.json", allow)
    man = template(allow)
    for ent in man["translations"]:
        ent["target"] = EN[ent["source"]]
        if ent["format_slots"]:
            ent["format_changes"] = {"latin": "Segoe UI", "lang": "en-US"}
            ent["reason"] = "Synthetic English prose; preserve source point size and emphasis."
    for m in man["visual_reviews"]:
        require(m["kind"] == "visual", "Unexpected held fixture object")
        m.update(
            decision="approved-logo",
            reason="Synthetic Japanese bitmap logo, deliberately preserved; compare on page 1.",
        )
    new_json(work / "manifest.json", man)
    apply(src, dst, allow, man)
    dll = validator(args.dotnet, work)
    report = compare(src, dst, allow, man, args.dotnet, dll)
    new_json(work / "comparison.json", report)
    require(report["status"] == "PASS", str(report["new_sdk_errors"]))
    require(
        not report["source_sdk"]["errors"],
        "Synthetic source has SDK errors: " + str(report["source_sdk"]["errors"][:3]),
    )
    result.update(
        sdk=report["source_sdk"]["sdk"],
        sdk_multiplicity_new_errors=len(report["new_sdk_errors"]),
        original_sha256=sourcehash,
        english_sha256=file_hash(dst),
        validator_sha256=file_hash(dll),
    )
    from text_ranges import protected, preserve_literals

    for token, wrong in (
        ("v7.2.0", "v8.2.0"),
        ("./a.out", "./b.out"),
        ("hipGetDeviceCount", "hipGetDevice"),
        ("0.4 GHz", "0.4 MHz"),
        ("https://example.org/v1", "https://example.org/v2"),
        ("hipcc -c file.hip", "hipcc -o file.hip"),
        ("userApi", "otherApi"),
    ):
        require(
            protected(token) != protected(wrong) or not preserve_literals(token, wrong),
            "Unprotected technical pair: " + token,
        )
    result["negative_checks"]["technical literal classes"] = "all changed examples identified"
    tampered = deepcopy(allow)
    tampered["ranges"][next(iter(tampered["ranges"]))]["xpath"] = "/p:sld"
    rejection(
        result,
        "forged allowlist",
        lambda: apply(src, work / "should-not-exist.pptx", tampered, man),
        "Allowlist differs",
    )
    retained = deepcopy(man)
    retained["translations"][0].update(
        target=retained["translations"][0]["source"], keep_reason="self-approved", format_changes={}
    )
    rejection(
        result,
        "unapproved retention",
        lambda: apply(src, work / "should-not-exist.pptx", allow, retained),
        "independent owner exception",
    )
    bad = deepcopy(man)
    bad["translations"][0]["target"] += " 200 GHz"
    rejection(
        result,
        "protected token",
        lambda: apply(src, work / "should-not-exist.pptx", allow, bad),
        "Protected token",
    )
    unlisted = work / "unauthorized.pptx"
    bad_copy(dst, unlisted, pkg.slides[0], lambda r: r.set("showMasterSp", "0"))
    rejection(
        result,
        "unlisted XPath",
        lambda: compare(src, unlisted, allow, man, args.dotnet, dll),
        "Unlisted XPath",
    )
    extra = work / "non-body-notes.pptx"

    def change_note(r):
        shape = first(r, ".//p:sp[p:nvSpPr/p:nvPr/p:ph[@type='body']]")
        first(shape, ".//p:ph").set("type", "ftr")

    bad_copy(src, extra, "ppt/notesSlides/notesSlide1.xml", change_note)
    rejection(
        result,
        "user text outside notes body",
        lambda: require(not inventory(Package(extra), [1, 2])[0]["blocking"], "non-body notes"),
        "non-body notes",
    )
    broken = work / "broken-reference.pptx"
    bad_copy(
        src,
        broken,
        "ppt/slides/_rels/slide1.xml.rels",
        lambda r: r[0].set("Target", "../no-such-part.xml"),
    )
    rejection(
        result, "unresolved OPC relationship", lambda: inventory(Package(broken), [1, 2]), "target"
    )
    sdk_base = report["source_sdk"]
    sdk_mutant = deepcopy(sdk_base)
    example = dict(
        id="X",
        type="Schema",
        part="/ppt/slides/slide1.xml",
        path="/p:sld",
        related=None,
        description="test",
    )
    old = dict(sdk_base, errors=[example])
    sdk_mutant["errors"] = [dict(example, path="/p:sld/p:cSld")]
    require(
        len(compare_sdk(old, sdk_mutant)) == 1, "Equal SDK counts must not mask different occurrences"
    )
    result["negative_checks"]["different SDK error with equal count"] = "rejected by multiset"
    invalid = work / "schema-invalid.pptx"
    bad_copy(src, invalid, pkg.slides[0], lambda r: first(r, ".//a:rPr").set("sz", "0"))
    errors = compare_sdk(sdk_base, sdk_report(args.dotnet, dll, invalid))
    require(errors, "Official SDK must diagnose invalid DrawingML size")
    result["negative_checks"]["official SDK schema regression"] = errors
    trial = work / "atomic"
    trial.mkdir()
    tmpout = trial / "render_check.pdf"
    job = render_guard.prepare(src, tmpout)
    try:
        rejection(
            result, "global lock", lambda: render_guard.prepare(src, trial / "other.pdf"), "lock busy"
        )
        competing = subprocess.run(
            [
                sys.executable,
                "-B",
                str(HERE / "render_guard.py"),
                "prepare",
                str(src),
                str(trial / "other.pdf"),
            ],
            capture_output=True,
            text=True,
        )
        require(
            competing.returncode != 0 and "lock busy" in competing.stderr,
            "A second process entered the locked render session",
        )
        result["negative_checks"]["second process"] = "refused before Office invocation"
        (job / (job.name + ".pdf")).write_bytes(b"truncated")
        rejection(result, "invalid PDF", lambda: render_guard.publish(job), "temporary PDF")
        require(not tmpout.exists(), "Invalid result was published")
    finally:
        render_guard.finish(job)
    # Exercise the failure path with a prior, hash-owned output. This is a byte
    # sentinel for rollback testing, not a page rendered for visual acceptance.
    tmpout.write_bytes(b"prior-output-sentinel")
    previous_hash = file_hash(tmpout)
    proof = render_guard.stamp(tmpout, previous_hash)
    new_json(proof, {"output": str(tmpout), "pdf_sha256": previous_hash})
    proof_hash = file_hash(proof)
    job = render_guard.prepare(src, tmpout)
    try:
        (job / (job.name + ".pdf")).write_bytes(b"broken")
        rejection(result, "prior output retained", lambda: render_guard.publish(job), "temporary PDF")
        require(
            file_hash(tmpout) == previous_hash and file_hash(proof) == proof_hash,
            "Failed render modified previous PDF or proof",
        )
    finally:
        render_guard.finish(job)
    boxed_copy(dst, box, [1, 2])
    result["static"] = "PASS"
    if args.platform == "static":
        result["macos"] = "NOT_RUN"
        return
    require(sys.platform == "darwin", "macOS PowerPoint required")
    app = Path("/Applications/Microsoft PowerPoint.app")
    require(app.is_dir(), "Microsoft PowerPoint is not installed")
    result["host"] = {
        "os": platform.mac_ver()[0],
        "powerpoint_version": command(
            [
                "/usr/libexec/PlistBuddy",
                "-c",
                "Print :CFBundleShortVersionString",
                app / "Contents/Info.plist",
            ]
        ),
        "powerpoint_build": command(
            ["/usr/libexec/PlistBuddy", "-c", "Print :CFBundleVersion", app / "Contents/Info.plist"]
        ),
    }
    source = HERE.parent / "macos/render.applescript"
    compiled = work / "render.scpt"
    command(["/usr/bin/osacompile", "-o", compiled, source])
    result["render_script_sha256"] = file_hash(source)
    result["compiled_sha256"] = file_hash(compiled)
    for label, f in (("ja", src), ("en", dst), ("boxed", box)):
        pdf = f.with_suffix(".pdf")
        argv = ["/usr/bin/osascript", compiled, f, pdf, sys.executable, HERE / "render_guard.py"]
        if label == "ja":
            argv.append("font-probe")
        result["macos"] = "ATTEMPTED_" + label.upper()
        command(argv)
        result.setdefault("renders", {})[label] = {
            "sha256": file_hash(pdf),
            "command": [str(x) for x in argv],
            "pages": 2,
        }
        command(["pdftoppm", "-r", "300", "-png", pdf, f.parent / "page"])
    require(file_hash(src) == sourcehash, "Fixture original changed")
    r = audit_pdf(dst, dst.with_suffix(".pdf"), man, report, src.with_suffix(".pdf"), True)
    new_json(work / "pdf-audit.json", r)
    result["macos"] = "RENDERED"
    result["font_audit"] = r["status"]
    require(r["status"] == "PASS", "Font verification needs attention; see pdf-audit.json")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--platform", choices=("static", "macos"), required=True)
    cli.add_argument("--dotnet", required=True)
    cli.add_argument("--work", required=True)
    a = cli.parse_args()
    if a.platform == "macos":
        cli.error("Synthetic fixture is not PowerPoint-qualified; use scripts/macos/render.py with a known-good local PPTX")
    w = local_path(a.work)
    w.mkdir(parents=True, exist_ok=False)
    report = {
        "static": "NOT_RUN",
        "macos": "NOT_RUN",
        "windows_com": "NOT_RUN",
        "negative_checks": {},
    }
    try:
        run(a, w, report)
    except Exception as e:
        report["status"] = "STOPPED"
        report["reason"] = str(e)
        raise
    else:
        report["status"] = (
            "PASS" if a.platform == "static" else "AUTOMATED_PASS_VISUAL_REVIEW_PENDING"
        )
    finally:
        new_json(w / "verification.json", report)
    print(w / "verification.json")


if __name__ == "__main__":
    main()
