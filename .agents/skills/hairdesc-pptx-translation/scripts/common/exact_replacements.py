"""Owner-approved scalar edits, bound to a source package and an independent bilingual manuscript."""
from collections import Counter, defaultdict
import json
import re

from office_package import AuditError, JA, address, digest, local_path, q, require, xp
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
POLICIES = {
    "standard_translation": {"prose", "annotation"},
    "preserve_monospace": {"code_comment", "fortran_comment", "explanatory_label", "annotation"},
}
FORTRAN_SENTINEL = re.compile(r"^[ \t]*(?:[!c*]\$|![a-z][a-z0-9_]*[$*])", re.I | re.ASCII)
LINE_BREAK = re.compile(r"(\r\n|[\r\n\v])")


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


def scan_fortran_line(text, quote):
    """Return the ordinary ! offset, continued quote, and Japanese positions inside literals."""
    stripped = text.lstrip(" \t")
    literal_positions = set()
    if FORTRAN_SENTINEL.match(text) or not stripped:
        return None, quote, literal_positions
    if quote and stripped.startswith("!"):
        # A comment-only line can intervene between two parts of a character literal.
        return len(text) - len(stripped), quote, literal_positions
    i = 0
    while i < len(text):
        char = text[i]
        if quote:
            if JA.match(char):
                literal_positions.add(i)
            if char == quote:
                if text[i + 1:i + 2] == quote:
                    i += 2
                    continue
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "!":
            return i, None, literal_positions
        i += 1
    return None, quote if text.rstrip(" \t").endswith("&") else None, literal_positions


def fortran_line_context(text, state):
    quote, continued = state
    stripped = text.lstrip(" \t")
    start, next_quote, literals = scan_fortran_line(text, quote)
    if not stripped or stripped.startswith("!") or FORTRAN_SENTINEL.match(text):
        return start, (next_quote, continued), literals
    require(not stripped.startswith("&") or continued, "Fortran continuation has no preceding line")
    require(not quote or stripped.startswith("&"), "Missing & on a continued Fortran character literal")
    return start, (next_quote, text[:start].rstrip(" \t").endswith("&")), literals


def fortran_initial_state(pkg, row, source):
    """Recover lexical state from the same text body, including paragraphs and manual breaks."""
    node, = xp(pkg.xml[row["part"]], row["xpath"])
    body = node.getparent().getparent().getparent()
    state = (None, False)
    for para in body.findall(q("a:p")):
        lines = [[]]
        for child in para:
            if child.tag == q("a:r"):
                lines[-1].extend(child.findall(q("a:t")))
            elif child.tag == q("a:br"):
                lines.append([])
            elif child.tag == q("a:fld"):
                lines[-1].append(child)
        for line in lines:
            require(all(el.tag == q("a:t") for el in line), "Dynamic field in Fortran code")
            text = "".join(el.text or "" for el in line)
            if node in line:
                require(text == source, "Incomplete Fortran code line")
                return state
            for physical in LINE_BREAK.split(text)[::2]:
                _, state, _ = fortran_line_context(physical, state)
    raise AuditError("Fortran run is outside its text body")


def fortran_literal_check(source, target):
    # In ordinary prose this exact count phrase names the transform, not an identifier.
    # The original source and the registered target themselves remain unnormalized.
    if "FFTの本数" in source:
        target = re.sub(r"\b([Nn]umber of )FFTs\b", r"\1FFT", target)
    return preserve_literals(source, target)


def check_fortran_comment(pkg, row, line, translations, registered):
    source = "".join(value for _, value in line)
    state = fortran_initial_state(pkg, row, source)
    physical_lines = [[]]
    for key, old in line:
        edit = translations.get(key, {})
        new = edit.get("target", old)
        if old != new or edit.get("format_changes"):
            require(key in registered and old != new, "Unregistered edit in a Fortran code line")
        before, after = LINE_BREAK.split(old), LINE_BREAK.split(new)
        require(before[1::2] == after[1::2], "Fortran line breaks changed")
        for i, (a, b) in enumerate(zip(before[::2], after[::2])):
            if i:
                physical_lines.append([])
            physical_lines[-1].append((key, a, b))

    for parts in physical_lines:
        old = "".join(a for _, a, _ in parts)
        new = "".join(b for _, _, b in parts)
        start, state, literal_positions = fortran_line_context(old, state)
        offset = 0
        for key, a, b in parts:
            if key in registered:
                require(all(offset + m.start() in literal_positions and a[m.start():m.end()] == m.group()
                            for m in JA.finditer(b)), "Japanese remains outside an original Fortran literal")
            offset += len(a)
        if all(a == b for _, a, b in parts):
            continue
        require(start is not None, "Fortran edit is outside an ordinary comment")
        require(not FORTRAN_SENTINEL.match(new), "Translation would create a Fortran directive or sentinel")
        body = start + 1
        while body < len(old) and old[body] in " \t":
            body += 1
        require(old[:body] == new[:body] and not new[body:].startswith((" ", "\t")),
                "Fortran code, delimiter or comment spacing changed")
        require(old[len(old.rstrip(" \t")):] == new[len(new.rstrip(" \t")):],
                "Fortran trailing whitespace changed")
        protected_material(old[body:], new[body:])
        # Keep operators, including continuation &, even in explanatory text.
        operators = r"[!&<>=*/%|^~+#]|(?<![A-Za-z0-9])-|-(?![A-Za-z])"
        require(re.findall(operators, old[body:]) == re.findall(operators, new[body:]),
                "Fortran comment operator changed")
        offset = 0
        for _, a, b in parts:
            protected_end = max(0, min(len(a), body - offset))
            if a != b:
                require(a[:protected_end] == b[:protected_end] and JA.search(a[protected_end:])
                        and not JA.search(b[protected_end:])
                        and fortran_literal_check(a[protected_end:], b[protected_end:]),
                        "Only Japanese in the registered Fortran comment may change")
            offset += len(a)


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
    fortran = spec["source_role"] == "fortran_comment"
    require(
        row["source"] == spec["source_text"] and isinstance(target, str) and target.strip()
        and target == edit["target"] and (fortran or not JA.search(target)) and not edit.get("keep_reason")
        and isinstance(spec["reason"], str) and spec["reason"].strip(),
        "Exact source/target mismatch",
    )
    policy = spec["format_policy"]
    require(policy in POLICIES and spec["source_role"] in POLICIES[policy], "Invalid exact format policy or role")
    # The registration may add a term to translated prose. It cannot rewrite a
    # technical-only source run or remove an original literal from a mixed run.
    require(JA.search(row["source"]) or not re.search(r"[A-Za-z0-9_]", row["source"]),
            "Cannot translate an original identifier-only run")
    literal_check = fortran_literal_check if fortran else preserve_literals
    require(literal_check(row["source"], target), "Exact replacement changed an original identifier")
    if fortran:
        require(target != row["source"], "Fortran comment approval must translate Japanese text")
    require(not (lexemes(row["source"]) - lexemes(target)), "Exact replacement lost a technical token")
    mono = bool(MONO.search((row["format_before"] or {}).get("font") or ""))
    require(mono == (policy == "preserve_monospace"), "Exact format policy differs from the source font")
    if mono:
        require(not (set(edit["format_changes"]) - {"lang"}), "Monospace exact edit only permits language")


def check_unit(spec, row, refs, lines, translations, pkg, fortran_ids):
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
    require(not (FORTRAN_SENTINEL.match(old) or FORTRAN_SENTINEL.match(new)),
            "Fortran directive or conditional sentinel is protected")
    if spec["source_role"] == "fortran_comment":
        check_fortran_comment(pkg, row, matches[0], translations, fortran_ids)
        return old, new
    before = outside_comments(old)
    has_comment = any(token in ("//", "/**/") for token in before)
    if spec["source_role"] == "code_comment":
        require(has_comment and before == outside_comments(new), "Executable bytes or comment delimiters changed")
        require(preserve_literals(old, new), "Code comment changed an identifier")
    elif spec["format_policy"] == "preserve_monospace":
        bang, _, _ = scan_fortran_line(old, None)
        require(bang is None or not JA.search(old[bang + 1:]), "Use a Fortran comment approval")
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
    rules = doc["approved_exact_replacements"]
    for spec in rules:
        require(isinstance(spec, dict) and set(spec) == RULE_KEYS
                and isinstance(spec["range_id"], str), "Invalid exact replacement fields")
    fortran_ids = {spec["range_id"] for spec in rules if spec["source_role"] == "fortran_comment"}
    for spec in rules:
        key = spec["range_id"]
        require(isinstance(key, str) and re.fullmatch(r"[a-f0-9]{24}", key) and key in sources
                and key in edits and key not in exact, "Unknown, duplicate or unlisted exact range")
        row = sources[key]
        check_rule(spec, row, edits[key])
        paragraph = (row["part"], row["paragraph"])
        if paragraph not in lines:
            lines[paragraph] = source_lines(pkg, row, by_path)
        old, new = check_unit(spec, row, by_id[key], lines[paragraph], edits, pkg, fortran_ids)
        exact[key] = spec
        trail.append(dict(
            **spec, source_ascii=re.findall(r"[\x20-\x7e]+", row["source"]),
            target_ascii=re.findall(r"[\x20-\x7e]+", spec["target_text"]), unit_source=old, unit_target=new,
            technical_tokens=[list(t) for t in sorted(lexemes(old).elements())],
            commands=COMMAND.findall(old), quoted_literals=QUOTES.findall(old),
            source_operators=re.findall(r"[<>=*/&|^~+#]", old), target_operators=re.findall(r"[<>=*/&|^~+#]", new),
        ))
    return exact, trail
