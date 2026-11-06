"""Read-only OPC primitives shared by the editor and independent package audit."""

from collections import Counter, defaultdict, deque
from copy import copy
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import json
import os
import posixpath
import re
import sys
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile
from lxml import etree as ET

NS = dict(
    p="http://schemas.openxmlformats.org/presentationml/2006/main",
    a="http://schemas.openxmlformats.org/drawingml/2006/main",
    r="http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    c="http://schemas.openxmlformats.org/drawingml/2006/chart",
    dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram",
    dsp="http://schemas.microsoft.com/office/drawing/2008/diagram",
    mc="http://schemas.openxmlformats.org/markup-compatibility/2006",
    rel="http://schemas.openxmlformats.org/package/2006/relationships",
    ct="http://schemas.openxmlformats.org/package/2006/content-types",
)
JA = re.compile(r"[ぁ-ヿ㐀-鿿々〆ｦ-ﾟ]")


class AuditError(ValueError):
    pass


def require(test, message):
    if not test:
        raise AuditError(message)


def q(name):
    prefix, local = name.split(":")
    return "{%s}%s" % (NS[prefix], local)


def digest(data):
    return sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as f:
        return sha256(f.read()).hexdigest()


def xp(element, path):
    return element.xpath(path, namespaces=NS)


def first(element, path):
    nodes = xp(element, path)
    return nodes[0] if nodes else None


def address(node):
    """Stable, namespace-qualified XPath, independent of the producer's prefixes."""
    steps = []
    for cur in [node, *node.iterancestors()]:
        name = ET.QName(cur)
        prefix = next((p for p, uri in NS.items() if uri == name.namespace), None)
        if prefix:
            tag = prefix + ":" + name.localname
        else:
            uri = name.namespace or ""
            require("'" not in uri, "Unsupported namespace in path")
            tag = "*[namespace-uri()='%s' and local-name()='%s']" % (uri, name.localname)
        steps.append(
            "%s[%d]" % (tag, 1 + sum(1 for _ in cur.itersiblings(tag=cur.tag, preceding=True)))
        )
    return "/" + "/".join(reversed(steps))


def load_json(path):
    def pairs(items):
        d = dict(items)
        require(len(items) == len(d), "Duplicate JSON key")
        return d

    def invalid(v):
        raise AuditError("Invalid JSON constant " + v)

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def local_path(path):
    """Staging/output must be on the Mac system data volume or a local Windows disk."""
    path = Path(path).absolute()
    parent = path if path.exists() else next(p for p in path.parents if p.exists())
    require(
        not any(p.is_symlink() for p in (path, *path.parents) if str(p) not in ("/tmp", "/var")),
        "Use a physical local work directory: " + str(path),
    )
    if sys.platform == "darwin":
        require(
            parent.stat().st_dev == Path("/private/tmp").stat().st_dev,
            "Use a work directory on the Mac local system data volume",
        )
    elif os.name == "nt":
        import ctypes

        drive = Path(os.path.realpath(parent)).anchor
        require(
            not drive.startswith("\\\\")
            and ctypes.windll.kernel32.GetDriveTypeW(str(drive)) in (2, 3, 6),
            "Use a local Windows work disk",
        )
    else:
        raise AuditError("Supported work hosts: macOS and Windows")
    return path.resolve(strict=False)


def new_json(path, value):
    path = local_path(path)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def parse_xml(data):
    parser = ET.XMLParser(
        resolve_entities=False, load_dtd=False, no_network=True, remove_blank_text=False
    )
    root = ET.fromstring(data, parser)
    require(not root.getroottree().docinfo.doctype, "DTD not supported in PPTX parts")
    return root


def canonical(root):
    return ET.tostring(root.getroottree(), method="c14n", with_comments=True)


def rel_source(name):
    if name == "_rels/.rels":
        return ""
    folder, leaf = posixpath.split(name)
    require(posixpath.basename(folder) == "_rels", "Invalid relationship part " + name)
    return posixpath.join(posixpath.dirname(folder), leaf[:-5])


class Package:
    def __init__(self, path):
        self.path = Path(path)
        snapshot = self.path.read_bytes()
        self.sha = digest(snapshot)
        with ZipFile(BytesIO(snapshot)) as z:
            names = z.namelist()
            require(len(names) == len(set(names)), "Duplicate ZIP member")
            for n in names:
                require(
                    not n.startswith("/") and "\\" not in n and ".." not in PurePosixPath(n).parts,
                    "Non-canonical ZIP member " + n,
                )
            self.data = {n: z.read(n) for n in names if not n.endswith("/")}
        self.hashes = {p: digest(b) for p, b in self.data.items()}
        self.xml = {
            p: parse_xml(b) for p, b in self.data.items() if p.endswith((".xml", ".rels", ".vml"))
        }
        self.rels = defaultdict(dict)
        self.inbound = defaultdict(set)
        self.errors = []
        for p, root in self.xml.items():
            if not p.endswith(".rels"):
                continue
            owner = rel_source(p)
            if owner and owner not in self.data:
                self.errors.append("Missing relationship owner " + owner)
            for rel in root:
                rid, typ, target = (rel.get(k, "") for k in ("Id", "Type", "Target"))
                external = rel.get("TargetMode") == "External"
                if not external:
                    uri = urlsplit(target)
                    target = posixpath.normpath(
                        posixpath.join(posixpath.dirname(owner), unquote(uri.path))
                        if uri.path
                        else owner
                    ).lstrip("/")
                    if uri.scheme or uri.netloc or target not in self.data:
                        self.errors.append(
                            "Missing/invalid target: " + owner + "#" + rid + " -> " + target
                        )
                    self.inbound[target].add(owner)
                if not rid or rid in self.rels[owner]:
                    self.errors.append("Duplicate/missing relationship ID: " + owner + "#" + rid)
                self.rels[owner][rid] = {
                    "type": typ.rsplit("/", 1)[-1],
                    "target": target,
                    "external": external,
                }
        office = [
            v["target"]
            for v in self.rels[""].values()
            if v["type"] == "officeDocument" and not v["external"]
        ]
        require(len(office) == 1 and office[0] in self.xml, "Missing unique main presentation")
        self.main = office[0]
        self.slides = []
        for el in xp(self.xml[self.main], "./p:sldIdLst/p:sldId"):
            target = self.rels[self.main].get(el.get(q("r:id")), {}).get("target")
            require(
                target in self.xml and target not in self.slides,
                "Invalid presentation slide relationship",
            )
            self.slides.append(target)
        self.hidden = [
            i
            for i, s in enumerate(self.slides, 1)
            if self.xml[s].get("show", "1") in ("false", "0", "off")
        ]
        self._check_references()

    def _check_references(self):
        ct = self.xml["[Content_Types].xml"]
        overrides = {unquote(n.get("PartName", "")).lstrip("/") for n in xp(ct, "ct:Override")}
        exts = {n.get("Extension", "").lower() for n in xp(ct, "ct:Default")}
        self.errors.extend(
            "Missing content type target " + n for n in sorted(overrides - self.data.keys())
        )
        for n in self.data:
            if (
                n != "[Content_Types].xml"
                and n not in overrides
                and n.rsplit(".", 1)[-1].lower() not in exts
            ):
                self.errors.append("Missing content type: " + n)
        for part, root in self.xml.items():
            for el in root.iter():
                for attr, rid in el.attrib.items():
                    if attr.startswith("{" + NS["r"] + "}") and rid and rid not in self.rels[part]:
                        self.errors.append("Unknown relationship " + part + "#" + rid)
            if root.tag not in (q("p:sld"), q("p:notes")):
                continue
            ids = Counter(x.get("id") for x in xp(root, ".//p:cNvPr"))
            for key, count in ids.items():
                if count > 1:
                    # Choice and Fallback may carry the same logical drawing ID.
                    elements = [el for el in xp(root, ".//p:cNvPr") if el.get("id") == key]
                    branches = [
                        first(el, "ancestor::mc:Choice[1] | ancestor::mc:Fallback[1]")
                        for el in elements
                    ]
                    if (
                        any(b is None for b in branches)
                        or len(set(branches)) != len(branches)
                        or len({b.getparent() for b in branches}) != 1
                    ):
                        self.errors.append("Duplicate shape ID " + part + "#" + str(key))
            for el in xp(root, ".//a:stCxn | .//a:endCxn | .//p:spTgt"):
                key = el.get("spid") if el.tag == q("p:spTgt") else el.get("id")
                if key and key not in ids:
                    self.errors.append("Unresolved connector/animation " + part + "#" + key)

    def targets(self, part, kind):
        return [
            r["target"] for r in self.rels[part].values() if not r["external"] and r["type"] == kind
        ]

    def owners(self):
        owners = defaultdict(set)
        boundaries = {"slide", "slideLayout", "slideMaster", "notesMaster", "theme", "officeDocument"}
        for page, part in enumerate(self.slides, 1):
            queue = deque([part])
            while queue:
                item = queue.popleft()
                if page in owners[item]:
                    continue
                owners[item].add(page)
                queue.extend(
                    r["target"]
                    for r in self.rels[item].values()
                    if not r["external"] and r["type"] not in boundaries
                )
        return owners


def clone_zip(src, target, changed):
    """Copy raw part payloads; only the ZIP directory/compression may be rebuilt."""
    with ZipFile(src) as source, ZipFile(target, "x") as out:
        out.comment = source.comment
        for info in source.infolist():
            data = changed[info.filename] if info.filename in changed else source.read(info)
            out.writestr(copy(info), data)
