"""Independently undo only allowlisted scalar edits in memory, then compare the entire part."""

from office_package import canonical, parse_xml, q, require, xp


def masked_equal(original, candidate, part, allow, manifest):
    a = parse_xml(original)
    b = parse_xml(candidate)

    def pair(path):
        orig = xp(a, path)
        edited = xp(b, path)
        require(len(orig) == len(edited) == 1, "Non-unique audit XPath " + path)
        return orig[0], edited[0]

    sizes = {entry["id"]: entry["after"] for entry in manifest.get("layout_sizes", [])}
    for row in manifest["translations"]:
        if row["part"] != part:
            continue
        orig, edited = pair(row["xpath"])
        slot = row["slot"]
        if slot == "text":
            require(edited.text == row["target"], "Wrong translated value at " + row["id"])
            edited.text = orig.text
        else:
            require(edited.get(slot) == row["target"], "Wrong translated attribute")
            edited.set(slot, orig.get(slot))
        changes = dict(row["format_changes"])
        if row["id"] in sizes:
            changes["size_pt"] = sizes[row["id"]]
        if not changes:
            continue
        oldrp = orig.getparent().find(q("a:rPr"))
        rp = edited.getparent().find(q("a:rPr"))
        require(rp is not None, "Missing edited character properties")
        for key, attribute in (("size_pt", "sz"), ("lang", "lang"), ("latin", "typeface")):
            if key not in changes:
                continue
            old = oldrp
            new = rp
            if key == "latin":
                new = rp.find(q("a:latin"))
                old = oldrp.find(q("a:latin")) if oldrp is not None else None
            v = str(changes[key] * 100) if key == "size_pt" else changes[key]
            require(new is not None and new.get(attribute) == v, "Wrong edited format")
            if old is not None and attribute in old.attrib:
                new.set(attribute, old.get(attribute))
            else:
                new.attrib.pop(attribute)
            if key == "latin" and old is None and not len(new) and not new.attrib and not new.text:
                rp.remove(new)
        if oldrp is None and not len(rp) and not rp.attrib and not rp.text:
            edited.getparent().remove(rp)
    for adj in manifest["adjustments"]:
        item = allow["geometry"][adj["id"]]
        if item["part"] != part:
            continue
        orig, edited = pair(item["xpath"])
        attr = item["attribute"]
        require(edited.get(attr) == str(adj["after"]), "Wrong requested geometry")
        edited.set(attr, orig.get(attr))
    for item in manifest.get("frame_layout", []):
        if item["part"] != part:
            continue
        orig, edited = pair(item["xpath"])
        attr = item["attribute"]
        require(orig.get(attr) == item["before"] and edited.get(attr) == str(item["after"]),
                "Wrong requested text-frame layout")
        if item["before"] is None:
            edited.attrib.pop(attr)
        else:
            edited.set(attr, item["before"])
    require(canonical(a) == canonical(b), "Unlisted XPath change: " + part)
