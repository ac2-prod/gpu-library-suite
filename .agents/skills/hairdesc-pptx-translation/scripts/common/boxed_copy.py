"""QA-only stroke overlay on text boxes; never publish this file as an English deck."""

from office_package import ET, Package, clone_zip, first, local_path, parse_xml, q, require, xp


def boxed_copy(source, dest, pages):
    pkg = Package(source)
    changed = {}
    for page in pages:
        part = pkg.slides[page - 1]
        root = parse_xml(pkg.data[part])
        for sp in xp(root, ".//p:sp[p:txBody]/p:spPr"):
            line = first(sp, "./a:ln")
            if line is None:
                line = ET.Element(q("a:ln"))
                i = next(
                    (
                        i
                        for i, n in enumerate(sp)
                        if ET.QName(n).localname
                        in ("effectLst", "effectDag", "scene3d", "sp3d", "extLst")
                    ),
                    len(sp),
                )
                sp.insert(i, line)
            line.clear()
            line.set("w", "12700")
            f = ET.SubElement(line, q("a:solidFill"))
            ET.SubElement(f, q("a:srgbClr"), val="D90079")
        changed[part] = ET.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)
    dest = local_path(dest)
    require(
        dest.name == "render_check.pptx" and dest.parent.name == "boxed",
        "Only boxed/render_check.pptx is allowed",
    )
    clone_zip(source, dest, changed)


if __name__ == "__main__":
    import argparse
    from pptx_audit import pages_arg

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--slides", required=True)
    a = p.parse_args()
    boxed_copy(a.source, a.output, pages_arg(a.slides, len(Package(a.source).slides)))
