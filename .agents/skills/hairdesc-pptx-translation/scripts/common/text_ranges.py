"""Identify text, its drawing owner, formatting cascade and protected literals."""

from collections import Counter
import re
from office_package import ET, address, first, q, xp

SHAPES = {q("p:" + k) for k in ("sp", "pic", "cxnSp", "grpSp", "graphicFrame", "contentPart")} | {
    q("dsp:sp")
}
NONVISUAL = "./*/p:cNvPr | ./*/dsp:cNvPr"
TOKEN = re.compile(
    r"""(?x)
    https?://[A-Za-z0-9._~:/?\#\[\]@!$&'()*+,;=%-]+
  | (?<![A-Za-z0-9_])v?\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.-]+)?
  | (?:\.{1,2}/|/)[A-Za-z0-9_./+-]+
  | (?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_./+-]*\.(?:f90|f95|F90|c|cc|cu|cpp|hpp|h|hip|mod|o|out|exe|dll|dylib|xml|py|sh|json|pptx|pdf|so|a|txt))\b
  | --?[A-Za-z][A-Za-z0-9_:=.+/-]*
  | [A-Za-z_]\w*(?=\s*\()
  | \b(?:roc(?:fft|blas|sparse|solver|rand|thrust)|hip(?:fort|Memcpy|Malloc|Free)|cuda(?:Malloc|Memcpy)|cuFFT|cuBLAS)[A-Za-z0-9_]*
  | (?<![A-Za-z0-9_])(?-i:(?:hip|cuda|cu|roc|omp)[A-Z][A-Za-z0-9_]*)
  | [A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+
  | (?<![A-Za-z0-9_])[+-]?(?:\d+(?:\.\d+)*|\.\d+)(?:[eE][+-]?\d+)?(?:[ \t]*(?:[kMGT]?i?B(?:/s)?|ns|[µμu]s|ms|s|Hz|MHz|GHz|[GMKT]?FLOP/s|%)(?![A-Za-z]))?
  | (?<![A-Za-z0-9_])(?:GPU|CPU|HIP|ROCm|OpenMP|CUDA|FP32|FP64|MiB|GiB|GB|ms|MHz|GHz)(?![A-Za-z0-9_])
""",
    re.ASCII | re.IGNORECASE,
)
COMMAND = re.compile(
    r"(?m)^[ \t]*(?:\$\s+|(?:sudo|mpirun|amdflang|amdclang(?:\+\+)?|hipcc|nvfortran|nvcc|gcc|g\+\+|cmake|qsub)\s+)[^\n\r]*"
)
MONO = re.compile(r"mono|consolas|courier|menlo|monaco|code|等幅", re.I)


def protected(text):
    return sorted(
        [m.group() for m in TOKEN.finditer(text)]
        + ["CMD " + m.group().strip() for m in COMMAND.finditer(text)]
    )


def original_literals(text):
    # Also protect unrecognised API names, option arguments, extensions and
    # versioned sonames already present in a mixed Japanese/English range.
    return Counter(re.findall(r"[A-Za-z][A-Za-z0-9_]*(?:[.:/+~-][A-Za-z0-9_+-]+)*", text, re.ASCII))


def preserve_literals(source, target):
    expected = original_literals(source)
    actual = original_literals(target)
    return all(actual[key] == count for key, count in expected.items())


def text_path_safe(n, kind, shape):
    """A text-looking extension, field or animation is never an editing entrypoint."""
    parent = n.getparent()
    if n.tag != q("a:t") or parent is None or parent.tag != q("a:r"):
        return False
    para = parent.getparent()
    if para is None or para.tag != q("a:p"):
        return False
    body = para.getparent()
    if body is None or any(
        ET.QName(a).localname in ("extLst", "timing", "AlternateContent") for a in n.iterancestors()
    ):
        return False
    if kind == "chart":
        return body.tag == q("c:rich") and body.getparent().tag == q("c:tx")
    if kind == "smartart":
        return body.tag == q("dgm:t") and body.getparent().tag == q("dgm:pt")
    if shape is None:
        return False
    if shape.tag == q("p:sp"):
        return body.tag == q("p:txBody") and body.getparent() is shape
    return (
        shape.tag == q("p:graphicFrame")
        and body.tag == q("a:txBody")
        and body.getparent().tag == q("a:tc")
        and first(body, "ancestor::a:tbl") is not None
    )


def owner(node):
    return next((a for a in [node, *node.iterancestors()] if a.tag in SHAPES), None)


def shape_info(shape):
    if shape is None:
        return {}
    nv = first(shape, NONVISUAL)
    ph = first(shape, "./*/p:nvPr/p:ph")
    groups = []
    for parent in shape.iterancestors(q("p:grpSp")):
        g = first(parent, NONVISUAL)
        if g is not None:
            groups.append(g.get("id"))
    return {
        "kind": ET.QName(shape).localname,
        "id": nv.get("id") if nv is not None else None,
        "name": nv.get("name") if nv is not None else None,
        "groups": groups[::-1],
        "placeholder": ph.get("type", "obj") if ph is not None else None,
        "idx": ph.get("idx", "0") if ph is not None else None,
    }


def related_placeholder(pkg, part, info, typ):
    targets = pkg.targets(part, typ)
    if len(targets) != 1 or targets[0] not in pkg.xml:
        return None, None
    dest = targets[0]
    candidates = []
    for el in xp(pkg.xml[dest], ".//p:sp[p:nvSpPr/p:nvPr/p:ph]"):
        x = shape_info(el)
        if info.get("idx") is not None and info["idx"] == x["idx"]:
            candidates.append(el)
    if len(candidates) == 1:
        return dest, candidates[0]
    exact = [
        el
        for el in xp(pkg.xml[dest], ".//p:sp[p:nvSpPr/p:nvPr/p:ph]")
        if shape_info(el)["placeholder"] == info.get("placeholder")
    ]
    return dest, exact[0] if len(exact) == 1 else None


def note_kind(pkg, part, shape):
    if shape is None:
        return None
    ph = first(shape, "./*/p:nvPr/p:ph")
    if ph is None:
        return None
    if ph.get("type"):
        return ph.get("type")
    _, master = related_placeholder(pkg, part, shape_info(shape), "notesMaster")
    return shape_info(master).get("placeholder")


def style(pkg, part, node):
    run = node.getparent()
    para = next(node.iterancestors(q("a:p")), None)
    props = first(para, "./a:pPr") if para is not None else None
    level = int(props.get("lvl", "0")) + 1 if props is not None else 1
    level_tag = "a:lvl%dpPr/a:defRPr" % level
    shape = owner(node)
    info = shape_info(shape)
    chain = [
        ("run", first(run, "./a:rPr")),
        ("paragraph", first(props, "./a:defRPr") if props is not None else None),
    ]
    body = next(
        (
            a
            for a in node.iterancestors()
            if ET.QName(a).localname in ("txBody", "rich", "t") and a is not node
        ),
        None,
    )
    if body is not None:
        chain.append(("list", first(body, "./a:lstStyle/" + level_tag)))
    curpart = part
    current = info
    master = None
    for edge in (
        ("notesMaster",) if "/notesSlides/" in "/" + part else ("slideLayout", "slideMaster")
    ):
        newpart, p = related_placeholder(pkg, curpart, current, edge)
        if not newpart:
            break
        curpart = newpart
        if p is not None and current.get("idx") is not None:
            chain.append((edge, first(p, "./p:txBody/a:lstStyle/" + level_tag)))
            current = shape_info(p)
        if edge.endswith("Master"):
            master = pkg.xml[newpart]
    if master is not None:
        if master.tag == q("p:notesMaster"):
            path = "./p:notesStyle/"
        else:
            typ = current.get("placeholder")
            key = (
                "titleStyle"
                if typ in ("title", "ctrTitle")
                else "bodyStyle" if typ in ("body", "obj", "subTitle") else "otherStyle"
            )
            path = "./p:txStyles/p:" + key + "/"
        chain.append(("master-style", first(master, path + level_tag)))
    chain.append(("presentation", first(pkg.xml[pkg.main], "./p:defaultTextStyle/" + level_tag)))
    result = {"font": None, "size_pt": None, "bold": False, "italic": False, "sources": {}}
    for key, attr in (("size_pt", "sz"), ("bold", "b"), ("italic", "i"), ("font", "typeface")):
        for origin, el in chain:
            if el is None:
                continue
            item = first(el, "./a:latin") if key == "font" else el
            if item is not None and item.get(attr) is not None:
                v = item.get(attr)
                result[key] = (
                    int(v) / 100
                    if key == "size_pt"
                    else (v in ("1", "true", "on") if key in ("bold", "italic") else v)
                )
                result["sources"][key] = origin
                break
    theme = pkg.targets(curpart, "theme")
    if result["font"] in ("+mj-lt", "+mn-lt") and len(theme) == 1:
        typ = "majorFont" if result["font"] == "+mj-lt" else "minorFont"
        face = first(pkg.xml[theme[0]], ".//a:" + typ + "/a:latin")
        if face is not None:
            result["font"] = face.get("typeface")
            result["sources"]["theme"] = theme[0]
    if body is not None:
        auto = first(body, "./a:bodyPr/a:normAutofit")
        if auto is not None:
            result["autofit_scale"] = auto.get("fontScale", "100000")
    return result


def context(node):
    shape = owner(node)
    info = shape_info(shape)
    para = next(node.iterancestors(q("a:p")), None)
    cell = next(node.iterancestors(q("a:tc")), None)
    box = first(shape, "./p:spPr/a:xfrm | ./p:xfrm") if shape is not None else None
    bounds = None
    if box is not None and not info.get("groups") and box.get("rot", "0") == "0":
        off = first(box, "a:off")
        ext = first(box, "a:ext")
        if off is not None and ext is not None:
            bounds = [
                int(e.get(a)) / 12700 for e, a in ((off, "x"), (off, "y"), (ext, "cx"), (ext, "cy"))
            ]
    return {
        "shape": info,
        "paragraph": address(para) if para is not None else None,
        "cell": address(cell) if cell is not None else None,
        "bounds_pt": bounds,
    }


def note_outside_is_field(text, kind):
    if kind not in ("dt", "sldNum", "sldImg", "hdr", "ftr"):
        return False
    return any(a.tag == q("a:fld") for a in text.iterancestors()) or not (text.text or "").strip()
