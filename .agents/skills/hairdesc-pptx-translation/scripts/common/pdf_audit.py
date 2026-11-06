"""Associate actual PDF glyphs (PDF points) with each declared translation range."""

import argparse
import json
from pathlib import Path
import re
import unicodedata
import pdfplumber
from pypdf import PdfReader
from office_package import Package, digest, file_hash, load_json, new_json, require
from render_guard import stamp
from pptx_audit import inventory
from text_ranges import MONO


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace())


def facename(name):
    name = re.sub(r"^[A-Z]{6}\+", "", str(name).lstrip("/"))
    return re.sub(r"[^a-z0-9]", "", name.lower())


def page_fonts(page):
    """Record actual descendants/file streams, including fonts in Form XObjects."""
    fonts = {}
    seen = set()

    def walk(res):
        res = res.get_object()
        for ref in res.get("/Font", {}).get_object().values() if "/Font" in res else ():
            f = ref.get_object()
            for v in f.get("/DescendantFonts", []) or [f]:
                v = v.get_object()
                desc = v.get("/FontDescriptor")
                desc = desc.get_object() if desc is not None else {}
                name = str(v.get("/BaseFont", f.get("/BaseFont", ""))).lstrip("/")
                present = [
                    desc[k].get_object()
                    for k in ("/FontFile", "/FontFile2", "/FontFile3")
                    if k in desc
                ]
                embedded = any(
                    hasattr(stream, "get_data") and bool(stream.get_data()) for stream in present
                )
                names = {
                    facename(x)
                    for x in (name, f.get("/BaseFont", ""), desc.get("/FontName", ""))
                    if x
                }
                embedded = embedded and len(names) == 1
                obj = (getattr(ref, "idnum", id(f)), getattr(ref, "generation", 0))
                for alias in {
                    name,
                    str(f.get("/BaseFont", "")).lstrip("/"),
                    str(desc.get("/FontName", "")).lstrip("/"),
                }:
                    if alias:
                        fonts.setdefault(alias, {})[obj] = embedded
        for ref in res.get("/XObject", {}).get_object().values() if "/XObject" in res else ():
            key = (getattr(ref, "idnum", None), getattr(ref, "generation", None))
            if key in seen:
                continue
            seen.add(key)
            obj = ref.get_object()
            if "/Resources" in obj:
                walk(obj["/Resources"])

    walk(page["/Resources"])
    return fonts


def match_range(page, fonts, text, fmt, rect=None, footer=False):
    target = normalize(text)
    chars = page.chars
    if rect:
        x, y, w, h = rect
        chars = [
            c
            for c in chars
            if x - 3 <= (c["x0"] + c["x1"]) / 2 <= x + w + 3
            and y - 3 <= (c["top"] + c["bottom"]) / 2 <= y + h + 3
        ]
    glyphs = [(letter, c) for c in chars for letter in normalize(c["text"])]
    line = "".join(a for a, _ in glyphs)
    if not target or line.count(target) != 1:
        return {
            "status": "UNRESOLVED",
            "reason": "Cannot uniquely identify the text in the PowerPoint PDF",
            "matches": line.count(target) if target else 0,
        }
    pos = line.index(target)
    used = [c for _, c in glyphs[pos : pos + len(target)]]
    faces = sorted({str(c["fontname"]) for c in used})
    sizes = sorted({round(float(c["size"]), 2) for c in used})
    name = fmt.get("font") or ""
    pt = fmt.get("size_pt")
    suffix = ("bold" if fmt.get("bold") else "") + ("italic" if fmt.get("italic") else "")
    exp = facename(name)
    allowed = (
        {exp + suffix, exp + suffix + "mt", exp + suffix + "psmt"}
        if suffix
        else {exp, exp + "regular", exp + "mt", exp + "psmt"}
    )
    found = {facename(n) for n in faces}

    def embedded_resource(name):
        resources = fonts.get(name)
        if resources is None:
            resources = {
                obj: ok
                for alias, entries in fonts.items()
                if facename(alias) == facename(name)
                for obj, ok in entries.items()
            }
        return len(resources) == 1 and all(resources.values())

    embedded = bool(found) and all(embedded_resource(n) for n in faces)
    size_ok = pt is not None and all(
        abs(s - pt) <= 0.35 or ("autofit_scale" in fmt and 0 < s <= pt + 0.35) for s in sizes
    )
    min_ok = footer or MONO.search(name) or all(s >= 11.65 for s in sizes)
    status = "PASS" if name and found <= allowed and embedded and size_ok and min_ok else "UNRESOLVED"
    if name and not found <= allowed:
        status = "FONT_SUBSTITUTED"
    return dict(
        status=status,
        pdf_fonts=faces,
        embedded=embedded,
        pdf_size_pt=sizes,
        requested=fmt,
        bounding_box=[
            min(c["x0"] for c in used),
            min(c["top"] for c in used),
            max(c["x1"] for c in used),
            max(c["bottom"] for c in used),
        ],
    )


def render_record(pdf, source_hash, mode):
    checksum = file_hash(pdf)
    record = load_json(stamp(pdf, checksum))
    require(
        record["source_sha256"] == source_hash
        and record["pdf_sha256"] == checksum
        and record["output"] == str(Path(pdf).resolve())
        and record["mode"] == mode,
        "Wrong or stale PDF/source/render mode",
    )
    return record


def audit_pdf(pptx, pdf, manifest, comparison, original_pdf, probes=False):
    pkg = Package(pptx)
    _, allow = inventory(pkg, manifest["slides"])
    reader = PdfReader(pdf, strict=True)
    manifest_hash = digest(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode())
    require(
        comparison["status"] == "PASS"
        and comparison["english_sha256"] == pkg.sha
        and comparison["manifest_sha256"] == manifest_hash
        and comparison["source_sha256"] == manifest["source_sha256"]
        and comparison["slides"] == manifest["slides"],
        "Wrong comparison result or manifest",
    )
    en_record = render_record(pdf, pkg.sha, "normal")
    ja_record = render_record(
        original_pdf, manifest["source_sha256"], "font-probe" if probes else "normal"
    )
    require(
        all(
            en_record[k] == ja_record[k]
            for k in (
                "host_fingerprint",
                "office_version",
                "font_catalog_sha256",
                "expected_pages",
                "hidden_pages",
                "include_hidden",
            )
        ),
        "Original/English OS, PowerPoint, or page-selection mismatch",
    )
    require(len(reader.pages) == len(pkg.slides), "PDF excludes pages or contains extra pages")
    reports = []
    checks = []
    with pdfplumber.open(pdf) as doc:
        fonts = [page_fonts(p) for p in reader.pages]
        if probes:
            with pdfplumber.open(original_pdf) as original:
                original_fonts = page_fonts(PdfReader(original_pdf, strict=True).pages[0])
                for text, bold in (("QA FONT REGULAR", False), ("QA FONT BOLD", True)):
                    checks.append(
                        {
                            "text": text,
                            "pdf_sha256": ja_record["pdf_sha256"],
                            **match_range(
                                original.pages[0],
                                original_fonts,
                                text,
                                dict(font="Segoe UI", size_pt=20, bold=bold, italic=False),
                            ),
                        }
                    )
            require(
                "QA HIDDEN PAGE" in (doc.pages[-1].extract_text() or ""),
                "Hidden marker not on the expected PDF page",
            )
        for entry in manifest["translations"]:
            if entry["id"] not in comparison["translations"]:
                continue
            r = allow["ranges"][entry["id"]]
            fmt = r["format_before"]
            page = r["slide"]
            require(r["source"] == entry["target"], "Manifest does not describe this English file")
            if r["kind"] == "alt":
                res = {"status": "NOT_DISPLAYED", "reason": "Accessibility metadata"}
            elif r["kind"] == "notes":
                compliant = (
                    fmt
                    and fmt["font"]
                    and fmt["size_pt"]
                    and (
                        MONO.search(fmt["font"])
                        or (
                            fmt["font"] == "Segoe UI"
                            and fmt["size_pt"] >= 12
                            and float(fmt["size_pt"]).is_integer()
                        )
                    )
                )
                res = {"status": "OOXML_ONLY" if compliant else "UNRESOLVED", "format": fmt}
            elif fmt is None:
                res = {"status": "UNRESOLVED", "reason": "No effective format"}
            else:
                res = match_range(
                    doc.pages[page - 1],
                    fonts[page - 1],
                    r["source"],
                    fmt,
                    r["bounds_pt"],
                    r["shape"].get("placeholder") in ("ftr", "dt", "sldNum"),
                )
                if fmt["font"] != "Segoe UI" and not MONO.search(fmt["font"] or ""):
                    res["status"] = "NONCOMPLIANT_FONT"
            reports.append(
                {
                    "id": r["id"],
                    "slide": page,
                    "shape": r["shape"].get("id"),
                    "text": r["source"],
                    **res,
                }
            )
    ok = {"PASS", "OOXML_ONLY", "NOT_DISPLAYED"}
    require(
        sorted(r["id"] for r in reports) == sorted(comparison["translations"]),
        "Unaccounted translation ranges",
    )
    return {
        "status": "PASS" if all(x["status"] in ok for x in reports + checks) else "NEEDS_REVIEW",
        "method": "PowerPoint PDF glyphs and separate boxed visual review; not PowerPoint text-frame metrics",
        "independent_font_probes": checks,
        "ranges": reports,
        "pages": len(reader.pages),
        "hidden": pkg.hidden,
        "english_pdf_sha256": en_record["pdf_sha256"],
        "original_pdf_sha256": ja_record["pdf_sha256"],
        "manifest_sha256": manifest_hash,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("pptx", "pdf", "manifest", "comparison", "original-pdf", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--fixture-probes", action="store_true")
    a = p.parse_args()
    r = audit_pdf(
        a.pptx,
        a.pdf,
        load_json(a.manifest),
        load_json(a.comparison),
        a.original_pdf,
        a.fixture_probes,
    )
    new_json(a.out, r)
    require(r["status"] == "PASS", "Unverified/substituted PDF text; inspect " + a.out)
