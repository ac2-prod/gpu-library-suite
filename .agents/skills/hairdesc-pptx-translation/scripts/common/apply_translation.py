"""Sole PPTX writer. Derive authority from the source; the editable manifest grants none."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import os
from office_package import (
    AuditError,
    ET,
    JA,
    Package,
    clone_zip,
    digest,
    file_hash,
    load_json,
    local_path,
    parse_xml,
    q,
    require,
    xp,
)
from pptx_audit import inventory
from text_ranges import MONO, protected, preserve_literals
import exact_replacements


def change_format(n, row, changes, reason):
    require(
        isinstance(changes, dict) and not changes.keys() - set(row["format_slots"]),
        "Unsupported format change",
    )
    if not changes:
        return
    require(reason.strip(), "Missing reason for font/size change")
    old = row["format_before"]
    require(
        not MONO.search(old.get("font") or "") or not ({"latin", "size_pt"} & changes.keys()),
        "Preserve monospaced font and size",
    )
    run = n.getparent()
    rp = run.find(q("a:rPr"))
    if rp is None:
        rp = ET.Element(q("a:rPr"))
        run.insert(0, rp)
    if "latin" in changes:
        require(changes["latin"] == "Segoe UI", "English prose font must be Segoe UI")
        latin = rp.find(q("a:latin"))
        if latin is None:
            latin = ET.Element(q("a:latin"))
            # CT_TextCharacterProperties: latin precedes ea/cs/sym/hlinks/rtl/extLst.
            before = next(
                (
                    j
                    for j, c in enumerate(rp)
                    if ET.QName(c).localname
                    in ("ea", "cs", "sym", "hlinkClick", "hlinkMouseOver", "rtl", "extLst")
                ),
                len(rp),
            )
            rp.insert(before, latin)
        latin.set("typeface", "Segoe UI")
    if "size_pt" in changes:
        size = changes["size_pt"]
        footer = row["shape"].get("placeholder") in ("ftr", "dt", "sldNum")
        require(type(size) is int and (1 if footer else 12) <= size <= 400, "Invalid font size")
        rp.set("sz", str(size * 100))
    if "lang" in changes:
        require(changes["lang"] in ("en-US", "en-GB"), "Invalid English language tag")
        rp.set("lang", changes["lang"])


def expected(pkg, allow, manifest, exceptions=None, exact_inputs=None):
    pages = allow["slides"]
    require(
        pages == sorted(set(pages))
        and all(type(x) is int and 1 <= x <= len(pkg.slides) for x in pages),
        "Invalid allowlist slides",
    )
    base, authority = inventory(pkg, pages)
    require(not base["blocking"], str(base["blocking"]))
    require(allow == authority, "Allowlist differs from fresh source inventory")
    require(
        manifest.get("source_sha256") == pkg.sha and manifest.get("slides") == pages,
        "Manifest source/selection mismatch",
    )
    rootmap = {}
    used = set()
    translated = []
    exact, _ = exact_replacements.validate(pkg, authority, manifest, exact_inputs)

    def target(row):
        p = row["part"]
        if p not in rootmap:
            rootmap[p] = parse_xml(pkg.data[p])
        nodes = xp(rootmap[p], row["xpath"])
        require(len(nodes) == 1, "Non-unique XPath")
        return nodes[0]

    for entry in manifest["translations"]:
        key = entry["id"]
        require(key in authority["ranges"] and key not in used, "Unknown or repeated range " + key)
        used.add(key)
        r = authority["ranges"][key]
        require(
            all(entry.get(k) == v for k, v in r.items()),
            "Manifest changed source/format metadata: " + key,
        )
        val = entry["target"]
        keep = entry.get("keep_reason", "")
        require(
            isinstance(val, str)
            and val.strip()
            and val.count("\n") == r["source"].count("\n")
            and val.count("\r") == r["source"].count("\r"),
            "Missing target or inserted paragraph at " + key,
        )
        if keep:
            require(
                (exceptions or {}).get(key) == keep
                and val == r["source"]
                and not entry.get("format_changes"),
                "Retention needs an independent owner exception for this exact range",
            )
        elif key not in exact:
            require(r["editable"], "Protected code/English-only range: " + key)
        # A checked Fortran comment run can also contain immutable Japanese in a code literal.
        require(not JA.search(val) or keep.strip()
                or exact.get(key, {}).get("source_role") == "fortran_comment",
                "Japanese remains without retention reason: " + key)
        if key not in exact:
            require(protected(val) == r["protected_tokens"], "Protected tokens changed: " + key)
            require(preserve_literals(r["source"], val), "Original technical literal changed: " + key)
        for extra in entry.get("extra_protected", []):
            require(
                isinstance(extra, str)
                and extra
                and r["source"].count(extra) > 0
                and r["source"].count(extra) == val.count(extra),
                "Explicit protected token changed",
            )
        if val == r["source"] and not entry["format_changes"]:
            continue
        n = target(r)
        if r["slot"] == "text":
            require(n.text == r["source"], "Old text mismatch")
            n.text = val
        else:
            require(n.get(r["slot"]) == r["source"], "Old attribute mismatch")
            n.set(r["slot"], val)
        change_format(n, r, entry["format_changes"], entry["reason"])
        translated.append(key)
    require(
        all(k in used or not JA.search(v["source"]) for k, v in authority["ranges"].items()),
        "Manifest omits translatable Japanese",
    )
    positions = set()
    editable_shapes = {
        (authority["ranges"][k]["part"], authority["ranges"][k]["shape"].get("id"))
        for k in translated
    }
    fixed_shapes = {
        (r["part"], r["shape_id"]) for r in exact.values() if r["format_policy"] == "preserve_monospace"
    }
    for adj in manifest["adjustments"]:
        k = adj["id"]
        require(k not in positions and k in authority["geometry"], "Unlisted geometry")
        positions.add(k)
        r = authority["geometry"][k]
        require(
            (r["part"], r["shape"].get("id")) not in fixed_shapes,
            "Monospace exact shape geometry must be preserved",
        )
        require(
            (r["part"], r["shape"]["id"]) in editable_shapes
            and adj["before"] == r["source"]
            and adj["reason"].strip(),
            "Unjustified reposition",
        )
        val = adj["after"]
        require(
            type(val) is int and (r["attribute"] not in ("cx", "cy") or val > 0),
            "Invalid DrawingML dimension",
        )
        target(r).set(r["attribute"], str(val))
    if len(manifest["visual_reviews"]) != len(authority["manual"]):
        raise AuditError("Incomplete visual review")
    reviews = {t["id"]: t for t in manifest["visual_reviews"]}
    for m in authority["manual"]:
        t = reviews.get(m["id"], {})
        require(
            all(t.get(k) == v for k, v in m.items()) and t.get("reason", "").strip(),
            "Missing visual/held-object evidence",
        )
        decision = t.get("decision")
        permitted = {
            "visual": {"already-english", "approved-logo", "owner-approved-retention"},
            "inherited": {
                "not-displayed",
                "already-english",
                "approved-logo",
                "owner-approved-retention",
            },
            "protected-text": {"owner-approved-retention"},
        }
        require(decision in permitted[m["kind"]], "Unresolved protected object " + m["id"])
        if decision == "owner-approved-retention":
            require(
                (exceptions or {}).get(m["id"]) == t["reason"],
                "Missing independent protected-object exception",
            )
    changed = {
        p: ET.tostring(root.getroottree(), encoding="UTF-8", xml_declaration=True, standalone=True)
        for p, root in rootmap.items()
    }
    return base, changed, translated


def apply(source, output, allow, manifest, exceptions=None, exact_inputs=None):
    source = local_path(source)
    pkg = Package(source)
    dest = local_path(output)
    require(
        dest.suffix.lower() == ".pptx" and not dest.exists() and dest != pkg.path.resolve(),
        "Output must be a new local PPTX",
    )
    _, changed, _ = expected(pkg, allow, manifest, exceptions, exact_inputs)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        test = Path(tmp) / "candidate.pptx"
        clone_zip(source, test, changed)
        result = Package(test)
        require(not result.errors, str(result.errors))
        require(
            result.hashes
            == {k: digest(changed[k]) if k in changed else v for k, v in pkg.hashes.items()},
            "Unexpected part change",
        )
        require(file_hash(source) == pkg.sha, "Source changed while editing")
        os.link(test, dest)
    return dest


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "output", "allowlist", "manifest"):
        cli.add_argument("--" + arg, required=True)
    cli.add_argument("--slides", required=True)
    cli.add_argument("--exceptions")
    exact_replacements.add_arguments(cli)
    from pptx_audit import checked_inputs

    a = cli.parse_args()
    allow, exceptions = checked_inputs(a)
    print(apply(a.source, a.output, allow, load_json(a.manifest), exceptions, exact_replacements.read_inputs(a)))
