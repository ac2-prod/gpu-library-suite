"""Owner-approved scalar edits, bound to a source package and an independent bilingual manuscript."""
from collections import Counter, defaultdict
import json
import re

from office_package import JA, address, digest, local_path, q, require, xp
from text_ranges import COMMAND, MONO, preserve_literals

# Protect technical material even when a registered prose translation introduces a word
# such as GPU. Word-internal hyphens in English prose are not command-line options.
LEXEMES = re.compile(r"""
    (?P<URL>https?://[A-Za-z0-9._~:/?\#\[\]@!$&'()*+,;=%-]+)
  | (?P<option>(?<![A-Za-z0-9_])--?[A-Za-z][\w=:./+\-]*)
  | (?P<path>(?<![\w/])(?:\.{1,2}/|/)(?:[A-Za-z0-9_.+-]+/)*[A-Za-z0-9_.+-]+)
  | (?P<filename>[A-Za-z_][\w.+/-]*\.(?:f90|f95|c|cc|cpp|cu|h|hip|hpp|o|a|sh|py|json|pptx|pdf|so|xml|out|txt)\b)
  | (?P<identifier>\b(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*|\b[A-Za-z]\w*_\w+\b|\b__[A-Za-z_]\w*\b)
  | (?P<api>\b(?:cuda|cufft|cublas|cusparse|cusolver|curand|rocfft|rocblas|rocsparse|rocsolver|rocrand|hip|omp)[A-Z_][A-Za-z0-9_]*)
  | (?P<number>(?<![A-Za-z0-9_])[+\-]?(?:\d+(?:\.\d+)*|\.\d+)(?:[eE][+\-]?\d+)?(?:[ \t]*(?:[kMGT]?i?B(?:/s)?|[µμu]s|ms|ns|s|GHz|MHz|Hz|[GMKT]?FLOP/s|%)(?![A-Za-z]))?)
""", re.ASCII | re.VERBOSE)
QUOTES = re.compile(r'"(?:\\.|[^"\\\n])*"|(?<!\w)\'(?:\\.|[^\'\\\n])+\'(?!\w)', re.ASCII)
LOC = ("slide", "part", "kind", "shape_id", "paragraph", "paragraph_index", "run_index", "xpath", "range_id")
RULE_KEYS = set(LOC) | {"source_text", "target_text", "format_policy", "source_role", "reason", "unit"}
POLICIES = {"standard_translation": {"prose", "annotation"}, "preserve_monospace": {"code_comment", "explanatory_label", "annotation"}}


def add_arguments(parser):
    parser.add_argument("--exact-approvals", help="Independent, source-bound exact replacement approvals")
    parser.add_argument("--manuscript", help="Local bilingual JSON referenced by those approvals")


def read_inputs(args):
    require(bool(args.exact_approvals) == bool(args.manuscript), "Supply both exact approvals and manuscript")
    if not args.exact_approvals:
        return None

    def unique(items):
        result = dict(items)
        require(len(result) == len(items), "Duplicate key in exact input")
        return result

    inputs = []
    for path in (args.exact_approvals, args.manuscript):
        data = local_path(path).read_bytes()
        document = json.loads(
            data.decode("utf-8"), object_pairs_hook=unique,
            parse_constant=lambda _: require(False, "Invalid JSON constant"),
        )
        inputs.extend((document, digest(data)))
    return tuple(inputs)


def location(row):
    """Both indices are 1-based, within the actual text body and paragraph respectively."""
    para = re.search(r"/a:p\[(\d+)\]$", row["paragraph"] or "")
    run = re.search(r"/a:r\[(\d+)\]/a:t\[1\]$", row["xpath"])
    require(para and run and row["slot"] == "text" and row["shape"]["id"], "Unaddressable exact run")
    return dict(
        slide=row["slide"], part=row["part"], kind=row["kind"], shape_id=row["shape"]["id"],
        paragraph=row["paragraph"], paragraph_index=int(para[1]), run_index=int(run[1]),
        xpath=row["xpath"], range_id=row["id"],
    )


def lexemes(text):
    return Counter((match.lastgroup, match.group().strip()) for match in LEXEMES.finditer(text))


def protected_material(source, target):
    require(lexemes(source) == lexemes(target), "Exact replacement changed a technical token")
    for pattern, label in ((COMMAND, "command"), (QUOTES, "quoted literal")):
        before = Counter(m.group() for m in pattern.finditer(source))
        after = Counter(m.group() for m in pattern.finditer(target))
        require(before == after, "Exact replacement changed a " + label)


def outside_comments(text):
    # Preserve quotes and executable bytes verbatim. Only complete comments are editable.
    parts = re.split(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|https?://[^\s"<>]+|//[^\n]*|/\*.*?\*/)', text, flags=re.S)
    return tuple(
        "//" if part.startswith("//") else "/**/" if part.startswith("/*") and part.endswith("*/") else part
        for part in parts
    )


def source_lines(pkg, row, ranges_by_xpath):
    matches = xp(pkg.xml[row["part"]], row["paragraph"])
    require(len(matches) == 1, "Ambiguous manuscript paragraph")
    lines = [[]]
    for child in matches[0]:
        if child.tag in (q("a:br"), q("a:fld")):
            lines.append([])
        elif child.tag == q("a:r"):
            for text in child.findall(q("a:t")):
                key = ranges_by_xpath.get((row["part"], address(text)))
                lines[-1].append((key, text.text or ""))
    return lines


def check_rule(spec, row, edit):
    loc = location(row)
    require(all(type(spec[k]) is type(v) and spec[k] == v for k, v in loc.items()), "Exact location mismatch")
    target = spec["target_text"]
    require(
        row["source"] == spec["source_text"] and isinstance(target, str) and target.strip()
        and target == edit["target"] and not JA.search(target) and not edit.get("keep_reason")
        and isinstance(spec["reason"], str) and spec["reason"].strip(),
        "Exact source/target mismatch",
    )
    policy = spec["format_policy"]
    require(policy in POLICIES and spec["source_role"] in POLICIES[policy], "Invalid exact format policy or role")
    # The registration may add a term to translated prose. It cannot rewrite a
    # technical-only source run or remove an original literal from a mixed run.
    require(JA.search(row["source"]) or not re.search(r"[A-Za-z0-9_]", row["source"]),
            "Cannot translate an original identifier-only run")
    require(preserve_literals(row["source"], target), "Exact replacement changed an original identifier")
    require(not (lexemes(row["source"]) - lexemes(target)), "Exact replacement lost a technical token")
    mono = bool(MONO.search((row["format_before"] or {}).get("font") or ""))
    require(mono == (policy == "preserve_monospace"), "Exact format policy differs from the source font")
    if mono:
        require(not (set(edit["format_changes"]) - {"lang"}), "Monospace exact edit only permits language")


def check_unit(spec, row, refs, lines, translations):
    require(len(refs) == 1, "Missing or ambiguous manuscript range")
    ref = refs[0]
    require(
        ref["unit"] == spec["unit"] and (ref["page"], ref["part"], ref["kind"], ref["shape"], ref["paragraph"])
        == (row["slide"], row["part"], row["kind"], row["shape"]["id"], row["paragraph"]),
        "Exact manuscript location mismatch",
    )
    ids = ref["range_ids"]
    matches = [line for line in lines if [key for key, _ in line if key is not None] == ids]
    require(len(matches) == 1, "Manuscript must identify one complete source segment")
    old = "".join(value for _, value in matches[0])
    new = "".join(translations[key]["target"] if key in translations else value for key, value in matches[0])
    require(old == ref["source"] and new == ref["translation"], "Exact text differs from bilingual manuscript")
    protected_material(old, new)
    before = outside_comments(old)
    has_comment = any(token in ("//", "/**/") for token in before)
    if spec["source_role"] == "code_comment":
        require(has_comment and before == outside_comments(new), "Executable bytes or comment delimiters changed")
        require(preserve_literals(old, new), "Code comment changed an identifier")
    elif spec["format_policy"] == "preserve_monospace":
        require(not re.search(r'[;={}#]|//|/\*|(?:^|\s)\$\s', old) and not COMMAND.search(old)
                and preserve_literals(old, new), "Executable text is not a monospaced explanation")
        require(re.findall(r"[<>=*/&|^~+#]", old) == re.findall(r"[<>=*/&|^~+#]", new),
                "Monospace explanation changed an operator")
    elif has_comment:
        require(before == outside_comments(new), "Executable bytes or comment delimiters changed")
    return old, new


def validate(pkg, allow, manifest, inputs):
    """Check all permissions before editing or comparing. Only returned keys receive an exception."""
    if inputs is None:
        require(not manifest.get("exact_approvals_sha256"), "Missing bound exact approvals")
        return {}, []
    doc, approval_hash, manuscript, manuscript_hash = inputs
    require(
        isinstance(doc, dict) and set(doc) == {"version", "source_sha256", "slides", "manuscript_sha256",
                                               "approved_exact_replacements"}
        and type(doc["version"]) is int and doc["version"] == 1 and doc["source_sha256"] == pkg.sha
        and doc["slides"] == allow["slides"] and doc["manuscript_sha256"] == manuscript_hash
        and manifest.get("exact_approvals_sha256") == approval_hash,
        "Exact approval provenance mismatch",
    )
    require(isinstance(manuscript, list) and isinstance(doc["approved_exact_replacements"], list),
            "Invalid exact inputs")
    sources = allow["ranges"]
    by_path = {(row["part"], row["xpath"]): key for key, row in sources.items() if row["slot"] == "text"}
    edits = {entry["id"]: entry for entry in manifest["translations"]}
    by_id = defaultdict(list)
    for unit in manuscript:
        if unit["page"] in allow["slides"]:
            for key in unit["range_ids"]:
                by_id[key].append(unit)

    exact = {}
    trail = []
    lines = {}
    for spec in doc["approved_exact_replacements"]:
        require(isinstance(spec, dict) and set(spec) == RULE_KEYS, "Invalid exact replacement fields")
        key = spec["range_id"]
        require(isinstance(key, str) and re.fullmatch(r"[a-f0-9]{24}", key) and key in sources
                and key in edits and key not in exact, "Unknown, duplicate or unlisted exact range")
        row = sources[key]
        check_rule(spec, row, edits[key])
        paragraph = (row["part"], row["paragraph"])
        if paragraph not in lines:
            lines[paragraph] = source_lines(pkg, row, by_path)
        old, new = check_unit(spec, row, by_id[key], lines[paragraph], edits)
        exact[key] = spec
        trail.append(dict(
            **spec, source_ascii=re.findall(r"[\x20-\x7e]+", row["source"]),
            target_ascii=re.findall(r"[\x20-\x7e]+", spec["target_text"]), unit_source=old, unit_target=new,
            technical_tokens=[list(t) for t in sorted(lexemes(old).elements())],
            commands=COMMAND.findall(old), quoted_literals=QUOTES.findall(old),
            source_operators=re.findall(r"[<>=*/&|^~+#]", old), target_operators=re.findall(r"[<>=*/&|^~+#]", new),
        ))
    return exact, trail
