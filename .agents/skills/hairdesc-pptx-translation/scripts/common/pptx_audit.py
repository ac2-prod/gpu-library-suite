"""Inventory and compare an immutable source, an allowlist, and an edited PPTX."""

import argparse
from collections import Counter, defaultdict
import json
import subprocess
import exact_replacements
from office_package import (
    ET,
    JA,
    Package,
    address,
    canonical,
    digest,
    file_hash,
    first,
    load_json,
    local_path,
    new_json,
    parse_xml,
    q,
    require,
    xp,
)
from text_ranges import (
    context,
    note_kind,
    note_outside_is_field,
    owner,
    protected,
    preserve_literals,
    shape_info,
    style,
    text_path_safe,
    MONO,
)


def pages_arg(value, count):
    if value == "all":
        return list(range(1, count + 1))
    pages = [int(p) for p in value.split(",")]
    require(
        pages and len(pages) == len(set(pages)) and all(1 <= p <= count for p in pages),
        "Invalid slide selection",
    )
    return sorted(pages)


def aux_kind(pkg, part):
    kinds = {
        rel["type"]
        for source in pkg.inbound[part]
        for rel in pkg.rels[source].values()
        if rel["target"] == part
    }
    if "chart" in kinds:
        return "chart"
    if "diagramData" in kinds:
        return "smartart"
    return "related"


def inventory(pkg, pages):
    require(not pkg.errors, "; ".join(pkg.errors))
    owners = pkg.owners()
    ranges = {}
    geometry = {}
    objects = []
    manual = []
    note_rows = []
    blocks = []
    note_assets = set()

    def descendants(starts):
        seen = set()
        pending = list(starts)
        while pending:
            part = pending.pop()
            if part in seen:
                continue
            seen.add(part)
            pending.extend(
                r["target"]
                for r in pkg.rels[part].values()
                if not r["external"] and r["type"] not in ("slide", "notesSlide", "officeDocument")
            )
        return seen

    shared_roots = {
        r["target"]
        for links in pkg.rels.values()
        for r in links.values()
        if r["type"] in ("slideLayout", "slideMaster", "theme", "notesMaster") and not r["external"]
    }
    global_parts = descendants(shared_roots)

    def add_manual(page, part, path, kind, detail):
        key = digest(json.dumps([page, part, path, kind], ensure_ascii=False).encode())[:24]
        if not any(r["id"] == key for r in manual):
            manual.append(dict(id=key, slide=page, part=part, xpath=path, kind=kind, detail=detail))

    def add_range(n, part, page, kind, slot="text"):
        value = n.text if slot == "text" else n.get(slot)
        if not value or not value.strip():
            return
        loc = address(n)
        key = digest((part + "#" + loc + "#" + slot).encode())[:24]
        ctx = context(n)
        if slot != "text":
            ctx["shape"] = shape_info(owner(n))
            ctx["paragraph"] = None
        item = dict(
            id=key,
            slide=page,
            part=part,
            xpath=loc,
            slot=slot,
            kind=kind,
            source=value,
            protected_tokens=protected(value),
            **ctx,
        )
        is_run = n.tag == q("a:t") and n.getparent().tag == q("a:r") and slot == "text"
        item["format_before"] = style(pkg, part, n) if is_run else None
        item["format_slots"] = ["latin", "size_pt", "lang"] if is_run else []
        font = (item["format_before"] or {}).get("font") or ""
        item["editable"] = bool(JA.search(value)) and not MONO.search(font)
        ranges[key] = item

    for i, slide in enumerate(pkg.slides, 1):
        notes = pkg.targets(slide, "notesSlide")
        require(len(notes) <= 1, "Multiple notes parts on slide " + str(i))
        if notes:
            part = notes[0]
            root = pkg.xml[part]
            para_count = 0
            fills = 0
            frames = []
            aux = descendants(
                r["target"]
                for r in pkg.rels[part].values()
                if not r["external"] and r["type"] not in ("notesMaster", "slide")
            )
            note_assets.update(aux)
            if aux:
                blocks.append(
                    f"p.{i}: classify non-body notes resources before editing: {sorted(aux)}"
                )
            for box in xp(root, ".//p:sp | .//p:graphicFrame | .//p:cxnSp | .//p:grpSp"):
                ps = [p for p in xp(box, ".//a:p") if owner(p) is box]
                if not ps:
                    continue
                kind = note_kind(pkg, part, box)
                texts = [t for p in ps for t in xp(p, ".//a:t")]
                nonempty = sum(
                    bool("".join(t.text or "" for t in xp(p, ".//a:t")).strip()) for p in ps
                )
                frames.append(
                    {
                        "shape": shape_info(box),
                        "kind": kind,
                        "paragraphs": len(ps),
                        "nonempty": nonempty,
                    }
                )
                if kind == "body":
                    para_count += len(ps)
                    fills += nonempty
                elif any(not note_outside_is_field(t, kind) for t in texts):
                    blocks.append(
                        f"p.{i} {part} {address(box)}: non-body notes text needs classification"
                    )
            for t in xp(root, ".//a:t"):
                if (t.text or "").strip() and owner(t) is None:
                    blocks.append(f"p.{i}: notes text without a known owner")
            note_rows.append(
                dict(
                    slide=i, part=part, body_paragraphs=para_count, body_nonempty=fills, frames=frames
                )
            )
        if i not in pages:
            continue
        for dep in sorted(descendants(pkg.targets(slide, "slideLayout"))):
            tree = pkg.xml.get(dep)
            if tree is None:
                continue
            if tree.tag not in (q("p:sldLayout"), q("p:sldMaster")):
                continue
            for n in xp(tree, ".//a:t | .//a:blip"):
                if n.tag == q("a:blip") or JA.search(n.text or ""):
                    add_manual(
                        i,
                        dep,
                        address(n),
                        "inherited",
                        {"origin": ET.QName(tree).localname, "shape": shape_info(owner(n))},
                    )
        for part, own in sorted(owners.items()):
            if i not in own:
                continue
            kind = "slide" if part == slide else "notes" if part in notes else aux_kind(pkg, part)
            if kind == "related":
                continue
            root = pkg.xml.get(part)
            if root is None:
                continue
            shared = (
                part in global_parts
                or part in note_assets
                or len(own) > 1
                or (
                    kind in ("chart", "smartart")
                    and any(s not in owners or owners[s] != own for s in pkg.inbound[part])
                )
            )
            for el in xp(
                root,
                ".//p:sp | .//p:grpSp | .//p:cxnSp | .//p:graphicFrame | .//p:pic | .//p:oleObj | .//p:timing | .//a:tbl | .//c:chart | .//dgm:relIds",
            ):
                obj = {"slide": i, "part": part, "xpath": address(el), "kind": ET.QName(el).localname}
                obj.update({k: v for k, v in shape_info(el).items() if k != "kind"})
                objects.append(obj)
                if el.tag in (q("p:pic"), q("p:oleObj")):
                    add_manual(
                        i, part, address(el), "visual", {"kind": obj["kind"], "shape": obj.get("id")}
                    )
            if kind == "slide":
                for blip in xp(root, ".//a:blip[not(ancestor::p:pic)]"):
                    add_manual(
                        i,
                        part,
                        address(blip),
                        "visual",
                        {"kind": "image-fill", "shape": shape_info(owner(blip)).get("id")},
                    )
                for box in xp(root, ".//p:sp"):
                    # Only existing, local text box attributes; never resize an autofit box or move a connector.
                    if (
                        first(box, "./p:txBody") is None
                        or first(box, "./p:txBody/a:bodyPr/a:normAutofit") is not None
                    ):
                        continue
                    for node, attrs in [
                        ("./p:spPr/a:xfrm/a:off", ("x", "y")),
                        ("./p:spPr/a:xfrm/a:ext", ("cx", "cy")),
                    ]:
                        e = first(box, node)
                        if e is None:
                            continue
                        for a in attrs:
                            if a not in e.attrib:
                                continue
                            loc = address(e)
                            gid = digest((part + "#" + loc + "@" + a).encode())[:24]
                            geometry[gid] = dict(
                                id=gid,
                                part=part,
                                slide=i,
                                shape=shape_info(box),
                                xpath=loc,
                                attribute=a,
                                source=e.get(a),
                            )
            for n in xp(root, ".//a:t | .//c:v"):
                if not (n.text or "").strip():
                    continue
                if kind == "notes" and note_kind(pkg, part, owner(n)) != "body":
                    continue
                if any(a.tag == q("a:fld") for a in n.iterancestors()):
                    continue
                if (
                    shared
                    or not text_path_safe(n, kind, owner(n))
                    or n.tag == q("c:v")
                    or (kind == "chart" and first(n, "ancestor::c:rich") is None)
                    or (
                        kind == "smartart"
                        and any(r["type"] == "diagramDrawing" for r in pkg.rels[part].values())
                    )
                ):
                    if JA.search(n.text):
                        add_manual(i, part, address(n), "protected-text", n.text)
                else:
                    add_range(n, part, i, kind)
            for n in xp(root, ".//p:cNvPr | .//dsp:cNvPr"):
                for attr in ("title", "descr"):
                    ownshape = owner(n)
                    if (
                        n.get(attr)
                        and not shared
                        and kind != "notes"
                        and ownshape is not None
                        and first(ownshape, "./*/p:cNvPr | ./*/dsp:cNvPr") is n
                        and not any(
                            ET.QName(a).localname in ("extLst", "timing", "AlternateContent")
                            for a in n.iterancestors()
                        )
                    ):
                        add_range(n, part, i, "alt", attr)
                    elif n.get(attr) and JA.search(n.get(attr)):
                        add_manual(i, part, address(n) + "/@" + attr, "protected-text", n.get(attr))
    # Report resources even when they are opaque or cannot safely be translated.
    for part, own in sorted(owners.items()):
        if not set(pages) & own:
            continue
        if (
            aux_kind(pkg, part) == "related"
            and part.endswith((".xml", ".vml"))
            and part not in pkg.slides
            and not any(x["part"] == part for x in note_rows)
        ):
            tree = pkg.xml[part]
            for n in tree.iter():
                if JA.search(n.text or "") and not len(n):
                    for i in sorted(set(pages) & own):
                        add_manual(i, part, address(n), "protected-text", n.text)
    summary = {
        "source_sha256": pkg.sha,
        "slides": pkg.slides,
        "selected": pages,
        "hidden": pkg.hidden,
        "pdf_includes_hidden": True,
        "notes": note_rows,
        "objects": objects,
        "manual": manual,
        "part_hashes": pkg.hashes,
        "relationships": dict(pkg.rels),
        "owners": {p: sorted(v) for p, v in owners.items()},
        "blocking": blocks,
    }
    allow = {
        "version": 1,
        "source_sha256": pkg.sha,
        "slides": pages,
        "ranges": ranges,
        "geometry": geometry,
        "manual": manual,
    }
    return summary, allow


def template(allow):
    return {
        "source_sha256": allow["source_sha256"],
        "slides": allow["slides"],
        "translations": [
            {**r, "target": None, "format_changes": {}, "reason": "", "keep_reason": ""}
            for r in allow["ranges"].values()
            if JA.search(r["source"])
        ],
        "adjustments": [],
        "visual_reviews": [{**r, "decision": None, "reason": ""} for r in allow["manual"]],
    }


def sdk_report(dotnet, dll, source):
    proc = subprocess.run([str(dotnet), str(dll), str(source)], text=True, capture_output=True)
    require(proc.returncode == 0, "OpenXmlValidator could not read package: " + proc.stderr)
    out = json.loads(proc.stdout)
    require(
        out.get("target") == "Microsoft365"
        and out.get("sdk")
        and isinstance(out.get("errors"), list),
        "Invalid SDK report",
    )
    return out


def compare_sdk(a, b):
    require(
        a["target"] == b["target"] and a["sdk"] == b["sdk"], "Different Open XML validation profiles"
    )

    def keys(report):
        return Counter(
            tuple(e.get(k) for k in ("id", "type", "part", "path", "related", "description"))
            for e in report["errors"]
        )

    return [{"identity": k, "new_count": v} for k, v in (keys(b) - keys(a)).items()]


def checked_inputs(args):
    allow = load_json(args.allowlist)
    src = Package(args.source)
    require(
        allow["slides"] == pages_arg(args.slides, len(src.slides)),
        "Requested page selection differs from allowlist",
    )
    if not args.exceptions:
        return allow, {}
    ex = load_json(args.exceptions)
    require(
        ex["source_sha256"] == src.sha and ex["slides"] == allow["slides"],
        "Exceptions belong to another source/scope",
    )
    return allow, ex["approved"]


def compare(original, candidate, allow, manifest, dotnet, dll, exceptions=None, exact_inputs=None):
    from apply_translation import expected
    from check_changes import masked_equal

    original = local_path(original)
    candidate = local_path(candidate)
    a = Package(original)
    b = Package(candidate)
    baseline, changed, translated = expected(a, allow, manifest, exceptions, exact_inputs)
    approved, trail = exact_replacements.validate(a, allow, manifest, exact_inputs)
    require(not b.errors, "; ".join(b.errors))
    require(a.data.keys() == b.data.keys(), "Package member set changed")
    for p in a.data:
        if p not in changed:
            require(a.hashes[p] == b.hashes[p], "Protected part changed: " + p)
        else:
            masked_equal(a.data[p], b.data[p], p, allow, manifest)
            require(
                canonical(parse_xml(changed[p])) == canonical(b.xml[p]), "Unlisted XPath change: " + p
            )
    after, new_allow = inventory(b, allow["slides"])
    require(not after["blocking"], str(after["blocking"]))
    require(baseline["notes"] == after["notes"], "Notes shapes/paragraph counts changed")
    exceptions = {r["id"] for r in manifest["translations"] if r.get("keep_reason")}
    # validate() established that each remaining Japanese character is in the original literal.
    literal_text = {k: rule["target_text"] for k, rule in approved.items()
                    if rule["source_role"] == "fortran_comment" and JA.search(rule["target_text"])}
    left = [
        {"id": k, "slide": v["slide"], "text": v["source"]}
        for k, v in new_allow["ranges"].items()
        if JA.search(v["source"]) and k not in exceptions
        and v["source"] != literal_text.get(k)
    ]
    require(not left, "Residual Japanese: " + str(left[:3]))

    # Explicit paragraph reconstruction also detects tokens spanning multiple formatted runs.
    def paragraphs(data, exact_before=None):
        groups = defaultdict(list)
        for r in data["ranges"].values():
            if r["paragraph"] and r["slot"] == "text":
                value = (exact_before or {}).get(r["id"], {}).get("target_text", r["source"])
                groups[(r["part"], r["paragraph"])].append(value)
        return {k: "".join(v) for k, v in groups.items()}

    originals, edits = paragraphs(allow, approved), paragraphs(new_allow)
    require(
        originals.keys() == edits.keys()
        and all(
            protected(v) == protected(edits[k]) and preserve_literals(v, edits[k])
            for k, v in originals.items()
        ),
        "Protected tokens changed across runs",
    )
    before_sdk = sdk_report(dotnet, dll, original)
    after_sdk = sdk_report(dotnet, dll, candidate)
    require(
        file_hash(original) == a.sha and file_hash(candidate) == b.sha,
        "Input changed during SDK validation",
    )
    added = compare_sdk(before_sdk, after_sdk)
    return dict(
        status="FAIL" if added else "PASS",
        source_sha256=a.sha,
        english_sha256=b.sha,
        changed_parts=sorted(changed),
        translations=translated,
        residual_japanese=left,
        slides=allow["slides"],
        manifest_sha256=digest(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()),
        validator_sha256=file_hash(dll),
        documented_exceptions=sorted(exceptions),
        approved_exact_replacements=trail,
        visual_reviews=manifest["visual_reviews"],
        source_sdk=before_sdk,
        english_sdk=after_sdk,
        new_sdk_errors=added,
        notes_part_count=len(after["notes"]),
        notes_with_body=sum(bool(x["body_nonempty"]) for x in after["notes"]),
        notes_body_paragraphs=sum(x["body_paragraphs"] for x in after["notes"]),
        hidden=b.hidden,
    )


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    sub = pa.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--source", required=True)
    inv.add_argument("--slides", default="all")
    inv.add_argument("--out", required=True)
    cmp = sub.add_parser("compare")
    for key in ("source", "english", "allowlist", "manifest", "dotnet", "validator", "out"):
        cmp.add_argument("--" + key, required=True)
    cmp.add_argument("--slides", required=True)
    cmp.add_argument("--exceptions")
    exact_replacements.add_arguments(cmp)
    a = pa.parse_args()
    if a.command == "inventory":
        pkg = Package(a.source)
        summary, allow = inventory(pkg, pages_arg(a.slides, len(pkg.slides)))
        out = local_path(a.out)
        out.mkdir(parents=True, exist_ok=False)
        for f, data in [("inventory", summary), ("allowlist", allow), ("manifest", template(allow))]:
            new_json(out / (f + ".json"), data)
        require(not summary["blocking"], str(summary["blocking"]))
        print(f"{len(allow['ranges'])} ranges; {len(summary['slides'])} slides; work {out}")
    else:
        allow, exceptions = checked_inputs(a)
        report = compare(
            a.source, a.english, allow, load_json(a.manifest), a.dotnet, a.validator, exceptions,
            exact_replacements.read_inputs(a)
        )
        new_json(a.out, report)
        require(
            report["status"] == "PASS", "New Open XML errors: " + str(report["new_sdk_errors"][:4])
        )
        print("PASS " + report["english_sha256"])


if __name__ == "__main__":
    main()
