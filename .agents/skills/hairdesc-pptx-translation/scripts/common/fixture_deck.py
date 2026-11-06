"""Synthetic two-slide QA package. Only the bundled blank Office template is used."""

from importlib.resources import files
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageDraw, ImageFont
from office_package import ET, NS, q, xp, parse_xml

EN = {
    "描画試験": "Rendering check",
    "準備 ": "Prepare ",
    "実行": "Execute",
    "ヘルプ": "Help",
    "入力": "Input",
    "出力": "Output",
    "設定を確認": "Check settings",
    "隠しページ": "Hidden page",
    "説明本文": "Speaker note",
    "確認事項": "Review item",
    "模式図の説明": "Diagram label",
}


def sub(parent, tag, **attrs):
    return ET.SubElement(parent, q(tag), {k: str(v) for k, v in attrs.items()})


def xml(root):
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)


def xfrm(shape, rect):
    node = sub(shape, "a:xfrm")
    x, y, w, h = [round(v * 12700) for v in rect]
    sub(node, "a:off", x=x, y=y)
    sub(node, "a:ext", cx=w, cy=h)


def textshape(tree, sid, name, rect, parts, *, auto=False, bullet=False, ph=None, field=False):
    shape = sub(tree, "p:sp")
    nv = sub(shape, "p:nvSpPr")
    sub(nv, "p:cNvPr", id=sid, name=name)
    sub(nv, "p:cNvSpPr", txBox="1")
    nvpr = sub(nv, "p:nvPr")
    if ph:
        sub(nvpr, "p:ph", type=ph[0], idx=ph[1])
    prop = sub(shape, "p:spPr")
    xfrm(prop, rect)
    g = sub(prop, "a:prstGeom", prst="rect")
    sub(g, "a:avLst")
    sub(prop, "a:noFill")
    tx = sub(shape, "p:txBody")
    bp = sub(tx, "a:bodyPr", wrap="square", lIns="38100", rIns="38100", tIns="12700", bIns="12700")
    if auto:
        sub(bp, "a:normAutofit")
    sub(tx, "a:lstStyle")
    p = sub(tx, "a:p")
    if bullet:
        pp = sub(p, "a:pPr", marL="342900", indent="-285750")
        sub(pp, "a:buChar", char="•")
    for v, pt, bold, link in parts:
        run = (
            sub(p, "a:fld", id="{FD878418-C421-47F0-AB01-ED43A9D76598}", type="slidenum")
            if field
            else sub(p, "a:r")
        )
        rp = sub(
            run, "a:rPr", lang="en-US" if v.isascii() else "ja-JP", sz=int(pt * 100), b=int(bold)
        )
        fill = sub(rp, "a:solidFill")
        sub(fill, "a:srgbClr", val="162641")
        sub(rp, "a:latin", typeface="Segoe UI" if v.isascii() else "Hiragino Sans")
        sub(rp, "a:ea", typeface="Hiragino Sans")
        if link:
            sub(rp, "a:hlinkClick").set(q("r:id"), "rIdLink")
        sub(run, "a:t").text = v
    return shape


def conn(tree):
    c = sub(tree, "p:cxnSp")
    nv = sub(c, "p:nvCxnSpPr")
    sub(nv, "p:cNvPr", id=11, name="Connected arrow")
    cp = sub(nv, "p:cNvCxnSpPr")
    sub(cp, "a:stCxn", id=8, idx=3)
    sub(cp, "a:endCxn", id=9, idx=1)
    sub(nv, "p:nvPr")
    sp = sub(c, "p:spPr")
    xfrm(sp, (164, 205, 110, 0))
    g = sub(sp, "a:prstGeom", prst="line")
    sub(g, "a:avLst")
    ln = sub(sp, "a:ln", w=25400)
    fill = sub(ln, "a:solidFill")
    sub(fill, "a:srgbClr", val="265477")
    sub(ln, "a:tailEnd", type="triangle")


def picture(tree):
    pic = sub(tree, "p:pic")
    nv = sub(pic, "p:nvPicPr")
    sub(nv, "p:cNvPr", id=12, name="Fixture logo", descr="模式図の説明")
    sub(nv, "p:cNvPicPr")
    sub(nv, "p:nvPr")
    fill = sub(pic, "p:blipFill")
    sub(fill, "a:blip").set(q("r:embed"), "rIdImage")
    sub(sub(fill, "a:stretch"), "a:fillRect")
    sp = sub(pic, "p:spPr")
    xfrm(sp, (44, 315, 130, 40))
    sub(sub(sp, "a:prstGeom", prst="rect"), "a:avLst")


def image_bytes():
    paths = [
        Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
        Path("C:/Windows/Fonts/meiryo.ttc"),
    ]
    path = next((p for p in paths if p.exists()), None)
    if not path:
        raise ValueError("A local Japanese font is required to generate the fixture bitmap")
    im = Image.new("RGB", (390, 120), "#e6f1f8")
    ImageDraw.Draw(im).text(
        (17, 14), "試験ロゴ", font=ImageFont.truetype(str(path), 58), fill="#163553"
    )
    b = BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def new_rels(items):
    el = ET.Element(q("rel:Relationships"), nsmap={None: NS["rel"]})
    for rid, typ, target in items:
        r = sub(el, "rel:Relationship", Id=rid, Type=NS["r"] + "/" + typ, Target=target)
        if typ == "hyperlink":
            r.set("TargetMode", "External")
    return xml(el)


def create_fixture(dest):
    t = files("pptx") / "templates"
    with ZipFile(t / "default.pptx") as z:
        data = {n: z.read(n) for n in z.namelist()}
    root = parse_xml(data["ppt/presentation.xml"])
    lst = ET.Element(q("p:sldIdLst"))
    nm = ET.Element(q("p:notesMasterIdLst"))
    sub(nm, "p:notesMasterId").set(q("r:id"), "rIdNotesM")
    root.insert(1, nm)
    root.insert(2, lst)
    presrels = parse_xml(data["ppt/_rels/presentation.xml.rels"])
    ct = parse_xml(data["[Content_Types].xml"])

    def override(part, kind):
        sub(
            ct,
            "ct:Override",
            PartName="/" + part,
            ContentType="application/vnd.openxmlformats-officedocument.presentationml."
            + kind
            + "+xml",
        )

    def mainrel(rid, kind, path):
        sub(presrels, "rel:Relationship", Id=rid, Type=NS["r"] + "/" + kind, Target=path)

    mainrel("rIdNotesM", "notesMaster", "notesMasters/notesMaster1.xml")
    data["ppt/notesMasters/notesMaster1.xml"] = (t / "notesMaster.xml").read_bytes()
    data["ppt/notesMasters/_rels/notesMaster1.xml.rels"] = new_rels(
        [("rId1", "theme", "../theme/theme1.xml")]
    )
    override("ppt/notesMasters/notesMaster1.xml", "notesMaster")
    for i in (1, 2):
        sl = ET.Element(q("p:sld"), nsmap={k: NS[k] for k in ("a", "r", "p")})
        if i == 2:
            sl.set("show", "0")
        template = parse_xml((t / "notes.xml").read_bytes())
        for ch in list(template):
            sl.append(ch)
        tree = xp(sl, "p:cSld/p:spTree")[0]

        def text(sid, name, rect, value, pt=18, bold=False, **kw):
            return textshape(tree, sid, name, rect, [(value, pt, bold, False)], **kw)

        if i == 1:
            text(2, "Title", (35, 23, 560, 48), "描画試験", 28, True)
            text(3, "QA-FontProbe-Regular", (38, 82, 480, 33), "QA FONT REGULAR", 20)
            text(4, "QA-FontProbe-Bold", (38, 119, 480, 33), "QA FONT BOLD", 20, True)
            textshape(
                tree,
                5,
                "Mixed text",
                (39, 152, 640, 35),
                [
                    ("準備 ", 18, False, False),
                    ("実行", 18, True, False),
                    ("  rocfft_execute()  4096 ms", 18, False, False),
                ],
            )
            text(8, "From", (43, 190, 121, 32), "入力")
            text(9, "To", (274, 190, 120, 32), "出力")
            conn(tree)
            text(13, "Autofit bullet", (40, 241, 430, 36), "設定を確認", 18, auto=True, bullet=True)
            textshape(tree, 14, "Link", (39, 278, 210, 30), [("ヘルプ", 16, False, True)])
            picture(tree)
        else:
            text(2, "Hidden heading", (38, 32, 560, 48), "隠しページ", 26, True)
            text(3, "Hidden marker", (38, 102, 540, 48), "QA HIDDEN PAGE", 20)
        file = f"ppt/slides/slide{i}.xml"
        data[file] = xml(sl)
        override(file, "slide")
        sub(lst, "p:sldId", id=255 + i).set(q("r:id"), f"rIdSlide{i}")
        mainrel(f"rIdSlide{i}", "slide", f"slides/slide{i}.xml")
        links = [
            ("rId1", "slideLayout", "../slideLayouts/slideLayout7.xml"),
            ("rIdNotes", "notesSlide", f"../notesSlides/notesSlide{i}.xml"),
        ]
        if i == 1:
            links.extend(
                [
                    ("rIdImage", "image", "../media/fixture.png"),
                    ("rIdLink", "hyperlink", "https://example.org/fixture"),
                ]
            )
        data[f"ppt/slides/_rels/slide{i}.xml.rels"] = new_rels(links)
        note = parse_xml((t / "notes.xml").read_bytes())
        nt = xp(note, "p:cSld/p:spTree")[0]
        textshape(
            nt,
            2,
            "Notes body",
            (30, 190, 600, 200),
            [("説明本文" if i == 1 else "確認事項", 16, False, False)],
            ph=("body", "3"),
        )
        textshape(
            nt,
            3,
            "Slide number",
            (600, 450, 40, 25),
            [(str(i), 12, False, False)],
            ph=("sldNum", "5"),
            field=True,
        )
        file = f"ppt/notesSlides/notesSlide{i}.xml"
        data[file] = xml(note)
        override(file, "notesSlide")
        data[f"ppt/notesSlides/_rels/notesSlide{i}.xml.rels"] = new_rels(
            [
                ("rId1", "notesMaster", "../notesMasters/notesMaster1.xml"),
                ("rId2", "slide", f"../slides/slide{i}.xml"),
            ]
        )
    if not xp(ct, "ct:Default[@Extension='png']"):
        sub(ct, "ct:Default", Extension="png", ContentType="image/png")
    data["ppt/media/fixture.png"] = image_bytes()
    data["[Content_Types].xml"] = xml(ct)
    data["ppt/presentation.xml"] = xml(root)
    data["ppt/_rels/presentation.xml.rels"] = xml(presrels)
    with ZipFile(dest, "x", ZIP_DEFLATED) as z:
        for k, v in data.items():
            z.writestr(k, v)
