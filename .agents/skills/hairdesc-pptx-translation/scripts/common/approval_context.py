"""Checks for explicitly addressed presentation prose, option labels, and font-only approvals."""
import re

from office_package import JA, q, require, xp
from text_ranges import COMMAND, MONO

# These are vetoes for a requested prose exception, never grounds for granting one.
CODE_LINE = re.compile(
    r"^[ \t]*(?:\$\s|#\s*(?:include|define|pragma)\b|!\$|(?:print\s*\*|write\s*\(|call\s+\w|"
    r"extern\s+[\"']|bind\s*\(|(?:if|while|for)\s*\(|(?:const|static|unsigned|struct|char|int|void|"
    r"double|float|integer|real|character)\b[^;\n]*(?:;|::|\w\s*[=(]))|[A-Za-z_]\w*\s*=(?!=)|"
    r"[A-Za-z_]\w*(?:::\w+)*\s*\([^;\n]*\)\s*[;{])",
    re.I | re.ASCII | re.M,
)
OPTION = re.compile(r"--?[A-Za-z][A-Za-z0-9_.+-]*(?:=[A-Za-z0-9_./+:-]+)?", re.ASCII)
LABEL = re.compile(r"^[ \t]*--?[A-Za-z][\w.+/=:-]*?(?:：|:(?=[ \t]))", re.ASCII)
ARGUMENT_PREFIX = re.compile(
    r"(?:[=/\\]|<<|--?[\w=.+/-]+|\b(?:extern|return|case)|\b\w+\s*\([^)]*|print\s*\*[^;]*)$",
    re.I | re.ASCII,
)
QUOTES = re.compile(r'"(?:\\.|[^"\\\n])*"|(?<!\w)\'(?:\\.|[^\'\\\n])+\'(?!\w)', re.ASCII)
PAIRS = {"「": "」", "『": "』", "“": "”", '"': '"'}
ROLES = {"prose", "option_description", "format_only", "code", "command"}


def pieces_at(line, start, end):
    pieces = []
    offset = 0
    for key, value in line:
        stop = offset + len(value)
        if start < stop and end > offset:
            pieces.append((key, value[max(0, start - offset):min(len(value), end - offset)]))
        offset = stop
    return pieces


def checked_spans(ref, old, new, line, after):
    spans = ref.get("protected_spans")
    require(isinstance(spans, list), "Specify this logical unit's protected spans")
    ends = [0, 0]
    for span in spans:
        require(isinstance(span, dict) and set(span) == {"source_start", "target_start", "text"},
                "Invalid protected span fields")
        a, b, text = span["source_start"], span["target_start"], span["text"]
        require(type(a) is int and type(b) is int and a >= ends[0] and b >= ends[1]
                and isinstance(text, str) and text and old[a:a+len(text)] == new[b:b+len(text)] == text,
                "Protected span position, order or text changed")
        ends = [a + len(text), b + len(text)]
        require(pieces_at(line, a, ends[0]) == pieces_at(after, b, ends[1]),
                "Protected span moved across PowerPoint runs")
    return spans


def prose_literal_text(ref, old, new, line, after, roles):
    """Mask the individually granted delimiters for the literal check only."""
    pairs = ref.get("prose_quotes", [])
    require(isinstance(pairs, list), "Invalid prose quote list")
    source, target = list(old), list(new)
    ends = [0, 0]
    for pair in pairs:
        keys = ("source_start", "source_end", "target_start", "target_end")
        require(isinstance(pair, dict) and set(pair) == set(keys), "Invalid prose quotation fields")
        a, z, b, y = (pair[k] for k in keys)
        require(all(type(i) is int for i in (a, z, b, y))
                and ends[0] <= a and a + 2 < z <= len(old) and ends[1] <= b and b + 2 < y <= len(new)
                and PAIRS.get(old[a]) == old[z-1] and new[b] == new[y-1] == '"',
                "Prose quote pair differs from the registered delimiters or positions")
        require(not any(ch in old[a+1:z-1] for ch in '「」『』“”"\r\n')
                and not any(ch in new[b+1:y-1] for ch in '"\r\n'),
                "Nested or multiline quotation needs separate evidence")
        for text, start, stop in ((old, a, z), (new, b, y)):
            prefix = text[:start].rstrip()
            inside = text[start+1:stop-1]
            require(not ARGUMENT_PREFIX.search(prefix)
                    and not re.match(r"(?:[/\\]|[.][A-Za-z0-9]+)", text[stop:])
                    and not re.match(r"(?:\.{0,2}/|~/|\$\{?\w+|[A-Za-z]:[\\/]|\\\\|--?[A-Za-z])", inside)
                    and not re.search(r"\.[A-Za-z0-9]{1,10}$", inside),
                    "Quoted command argument, option value, path or code literal is protected")
        overlaps = [m.span() for m in QUOTES.finditer(old) if m.start() < z and a < m.end()]
        if old[a] == '"':
            # A standalone ASCII literal is ambiguous. Require surrounding Japanese prose
            # as well as the explicit role and matched pair before translating its contents.
            require(overlaps == [(a, z)] and JA.search(old[:a] + old[z:]),
                    "Ambiguous existing ASCII string literal")
        else:
            require(not overlaps, "Cannot change delimiters inside an existing string literal")
        for s, t in ((a, b), (z-1, y-1)):
            src_key = pieces_at(line, s, s+1)[0][0]
            dst_key = pieces_at(after, t, t+1)[0][0]
            require(src_key == dst_key and roles.get(src_key) == "prose",
                    "Each prose delimiter needs an exact approval in its original run")
            source[s] = target[t] = "「" if s == a else "」"
        ends = [z, y]
    return "".join(source), "".join(target)


def option_suffixes(spans, old, new):
    start = len(old) - len(old.lstrip(" \t"))
    matches = [s["text"] for s in spans if s["source_start"] == s["target_start"] == start]
    require(len(matches) == 1 and OPTION.fullmatch(matches[0]), "Missing complete protected option token")
    end = start + len(matches[0])
    require(old[:end] == new[:end] and old[end:end+1] in ("：", ":") and new[end:end+1] == ":"
            and (old[end] != ":" or old[end+1:end+2] in (" ", "\t"))
            and new[end+1:end+2] in (" ", "\t") and old[end+1:].strip() and new[end+1:].strip(),
            "Option prefix, value or description delimiter changed")
    return old[end+1:], new[end+1:]


def check_font_context(spec, line, old, spans):
    start = 0
    for key, text in line:
        if key == spec["range_id"]:
            break
        start += len(text)
    end = start + len(spec["source_text"])
    protected = [(m.start(), m.end()) for m in QUOTES.finditer(old)]
    protected += [(s["source_start"], s["source_start"] + len(s["text"])) for s in spans]
    require(not any(start < stop and pos < end for pos, stop in protected)
            and not re.search(r'::|[;={}#]|!\$|(?:^|\s)--?[\w+]|(?:bind\s*\(\s*C\b|extern\s*")',
                              spec["source_text"], re.I | re.ASCII),
            "Format-only cannot restyle protected code, options or string literals")


def check_font_rule(pkg, row, edit, spec):
    changes = spec["target_format"]
    require(spec["format_policy"] == "format_only" and row["kind"] == "slide"
            and row["slide"] not in pkg.hidden and row["source"] == spec["source_text"] == spec["target_text"] == edit["target"]
            and not JA.search(row["source"]) and re.search("[A-Za-z]", row["source"])
            and not edit.get("keep_reason"), "Format-only requires unchanged visible English text")
    require(isinstance(changes, dict) and changes and changes == edit["format_changes"]
            and set(changes) <= {"latin", "lang"}
            and all(v == {"latin": "Segoe UI", "lang": "en-US"}[k] for k, v in changes.items()),
            "Format-only grant covers only the specified Latin font and language")
    font = row["format_before"]["font"]
    require(font and not MONO.search(font), "Format-only cannot restyle a code or unresolved font")
    t, = xp(pkg.xml[row["part"]], row["xpath"])
    rp = t.getparent().find(q("a:rPr"))
    before = {"latin": font, "lang": rp.get("lang") if rp is not None else None}
    require(spec["expected_before"] == {k: before[k] for k in changes}
            and any(before[k] != v for k, v in changes.items()), "Format-only source attributes do not match")
