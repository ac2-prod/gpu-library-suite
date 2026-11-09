"""Exercise the editor and audit with synthetic PowerPoint code lines; no PowerPoint process is used."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pptx import Presentation
from pptx.util import Inches, Pt

import exact_replacements as exact
from apply_translation import apply, expected
from office_package import AuditError, Package, address, q, xp
from pptx_audit import compare, inventory, template


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


class FortranCommentTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name).resolve()
        self.fixture_no = 0

    def exercise(self, lines, edits, *, omit=(), role="fortran_comment", soft=False, sdk=False):
        self.fixture_no += 1
        source = self.work / f"input-{self.fixture_no}.pptx"
        ppt = Presentation()
        slide = ppt.slides.add_slide(ppt.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(3))
        frame = box.text_frame
        para = frame.paragraphs[0]
        for i, values in enumerate(lines):
            if i and soft:
                para.add_line_break()
            elif i:
                para = frame.add_paragraph()
            for text in values:
                run = para.add_run()
                run.text = text
                run.font.name = "Consolas"
                run.font.size = Pt(15)
                run.font.bold = True
        ppt.save(source)
        pkg = Package(source)
        _, allow = inventory(pkg, [1])
        manuscript, rules = [], []
        manifest = template(allow)
        entries = {r["id"]: r for r in manifest["translations"]}
        ranges = {r["xpath"]: r for r in allow["ranges"].values() if r["slot"] == "text"}
        nodes = iter(xp(pkg.xml[pkg.slides[0]], ".//p:sp/p:txBody/a:p/a:r/a:t"))
        for i, values in enumerate(lines):
            ids = []
            targets = []
            for j, text in enumerate(values):
                node = next(nodes)
                self.assertEqual(node.text or "", text)
                row = ranges.get(address(node))
                target = edits.get((i, j), text)
                targets.append(target)
                if row is None:
                    self.assertEqual(text, target)
                    continue
                key = row["id"]
                ids.append(key)
                if (i, j) not in edits:
                    continue
                if key not in entries:
                    entry = dict(row, target=None, format_changes={}, keep_reason="", reason="")
                    manifest["translations"].append(entry)
                    entries[key] = entry
                entries[key].update(target=target, format_changes={"lang": "en-US"}, reason="Fixture")
                if (i, j) not in omit:
                    rules.append(dict(**exact.location(row), unit=str(i), source_text=text, target_text=target,
                                      source_role=role, format_policy="preserve_monospace", reason="Fixture"))
            if ids:
                first = allow["ranges"][ids[0]]
                manuscript.append(dict(unit=str(i), page=1, kind=first["kind"], part=first["part"],
                                       shape=first["shape"]["id"], paragraph=first["paragraph"], range_ids=ids,
                                       source="".join(values), translation="".join(targets)))
        approvals = dict(version=1, slides=[1], source_sha256=pkg.sha, manuscript_sha256=digest(manuscript),
                         approved_exact_replacements=rules)
        manifest["exact_approvals_sha256"] = digest(approvals)
        inputs = approvals, digest(approvals), manuscript, digest(manuscript)
        if not sdk:
            expected(pkg, allow, manifest, exact_inputs=inputs)
            return
        output = self.work / "validator-fixture.pptx"
        apply(source, output, allow, manifest, exact_inputs=inputs)
        report = compare(source, output, allow, manifest, os.environ["DOTNET_BIN"],
                         os.environ["VALIDATOR_DLL"], exact_inputs=inputs)
        self.assertEqual(report["status"], "PASS")
        for key in ("source_sdk", "english_sdk"):
            self.assertEqual(report[key]["errors"], [])
        self.assertEqual(report["new_sdk_errors"], [])
        self.assertEqual(report["residual_japanese"], [])
        self.assertEqual(len(report["approved_exact_replacements"]), len(edits))
        candidate = Package(output)
        for rule in rules:
            t, = xp(candidate.xml[rule["part"]], rule["xpath"])
            self.assertEqual(t.text, rule["target_text"])
            rp = t.getparent().find(q("a:rPr"))
            self.assertEqual((rp.get("lang"), rp.get("sz"), rp.get("b"), rp.find(q("a:latin")).get("typeface")),
                             ("en-US", "1500", "1", "Consolas"))

    def test_inline_comment(self):
        self.exercise([["integer :: n  ! 要素数"]], {(0, 0): "integer :: n  ! Number of elements"})

    def test_comment_only(self):
        self.exercise([["! 数値検証"]], {(0, 0): "! Numerical verification"})

    def test_fft_count_in_one_approved_run(self):
        self.exercise([["integer, parameter :: batch = 4096 ! FFTの本数"]],
                      {(0, 0): "integer, parameter :: batch = 4096 ! Number of FFTs"})

    def check_directive(self, name):
        # Exercise the lexical gate with a Japanese tail as well as an ASCII-only edit attempt.
        source = "\t  " + name + " data copyin(a)"
        self.assertIsNone(exact.scan_fortran_line(source, None)[0])
        with self.assertRaises(AuditError):
            self.exercise([[source]], {(0, 0): source.replace("copyin", "copyout")})
        with self.assertRaisesRegex(AuditError, "directive|sentinel|outside an ordinary comment"):
            self.exercise([[source + " ! 説明"]], {(0, 0): source + " ! Description"})

    def test_openacc_directive(self):
        self.check_directive("!$acc")

    def test_openmp_directive(self):
        self.check_directive("!$omp")

    def test_cuda_fortran_directive(self):
        self.check_directive("!$cuf")

    def test_conditional_and_fixed_form_sentinels(self):
        for sentinel in ("!$", "!$OMPx", "C$AcC", "*$acc", "c$omp", "*$OMP", "!DIR$"):
            with self.subTest(sentinel=sentinel):
                self.check_directive(sentinel)

    def test_directive_cannot_be_registered_as_label(self):
        with self.assertRaisesRegex(AuditError, "sentinel"):
            self.exercise([["!$acc data ! 説明"]], {(0, 0): "!$acc data ! Explanation"}, role="explanatory_label")

    def test_translation_cannot_create_a_directive(self):
        for new in ("!$omp parallel", "!DIR$ VECTOR"):
            with self.subTest(new=new), self.assertRaisesRegex(AuditError, "directive|sentinel"):
                self.exercise([["!説明"]], {(0, 0): new})
        with self.assertRaisesRegex(AuditError, "directive|sentinel"):
            self.exercise([["! 説明\n!注記"]], {(0, 0): "! Description\n!$acc parallel"})

    def check_literal(self, quote):
        source = "print *, " + quote + "value!" + quote
        self.assertIsNone(exact.scan_fortran_line(source, None)[0])
        with self.assertRaises(AuditError):
            self.exercise([[source]], {(0, 0): source.replace("!", "?")})
        text = source.replace("!", "!値")
        with self.assertRaises(AuditError):
            self.exercise([[text]], {(0, 0): text.replace("値", "value")})

    def test_double_quoted_bang(self):
        self.check_literal('"')

    def test_single_quoted_bang(self):
        self.check_literal("'")

    def test_doubled_quotes_in_either_kind_of_literal(self):
        for value in ("'don''t!change'", '"a""!b"'):
            prefix = f"print *, {value}  ! "
            with self.subTest(value=value):
                self.exercise([[prefix + "説明"]], {(0, 0): prefix + "Description"})

    def test_split_code_marker_and_comment(self):
        self.exercise([["integer, parameter :: nfft = 1024", "  ", "! FFT", "長"]], {(0, 3): " length"})

    def test_unregistered_comment(self):
        with self.assertRaisesRegex(AuditError, "Protected code/English-only"):
            self.exercise([["! 数値検証"]], {(0, 0): "! Verify"}, omit={(0, 0)})

    def test_unregistered_sibling_run(self):
        with self.assertRaisesRegex(AuditError, "Unregistered edit"):
            self.exercise([["!", "説明", "補足"]], {(0, 1): "Description", (0, 2): " note"}, omit={(0, 2)})

    def test_api_identifier_and_numeric_changes(self):
        original = "call cudaDeviceSynchronize(nfft, 1024)  ! 説明"
        for old, new in (("cudaDeviceSynchronize", "cudaFree"), ("nfft", "n"), ("1024", "2048")):
            with self.subTest(token=old), self.assertRaises(AuditError):
                self.exercise([[original]], {(0, 0): original.replace(old, new).replace("説明", "Description")})

    def test_code_delimiter_whitespace_and_continuation(self):
        original = "  call f(a, &  ! 説明  "
        good = original.replace("説明", "Description")
        self.exercise([[original]], {(0, 0): good})
        for bad in (good.lstrip(), good.rstrip(), good.replace(", &", ", "), good.replace("  !", " !"),
                    good.replace("! ", "!"), good.replace("! ", "!  "), good.replace("!", "//")):
            with self.subTest(bad=bad), self.assertRaises(AuditError):
                self.exercise([[original]], {(0, 0): bad})

    def test_operator_in_comment_is_preserved(self):
        with self.assertRaisesRegex(AuditError, "operator"):
            self.exercise([["! A & B の説明"]], {(0, 0): "! A and B explanation"})

    def test_comment_command_is_preserved(self):
        with self.assertRaisesRegex(AuditError, "command"):
            self.exercise([["! nvfortran a.o b.o ! 説明"]], {(0, 0): "! nvfortran b.o a.o ! Note"})

    def test_literal_continued_across_powerpoint_lines(self):
        for soft in (False, True):
            with self.subTest(soft=soft):
                prefix = "&!保持'  ! "
                lines = [["print *, 'keep&"], ["! intervening comment"], [prefix + "説明"]]
                self.exercise(lines, {(2, 0): prefix + "Description"}, soft=soft)
                with self.assertRaises(AuditError):
                    self.exercise(lines, {(2, 0): "&!changed'  ! Description"}, soft=soft)

    def test_excerpt_starting_mid_continuation_is_rejected(self):
        with self.assertRaisesRegex(AuditError, "no preceding line"):
            self.exercise([["&!要素数"]], {(0, 0): "&!Elements"})

    def test_continued_statement_with_inline_comments(self):
        self.exercise([["call f(a, & ! 第一引数"], ["  & b) ! 第二引数"]],
                      {(0, 0): "call f(a, & ! First argument", (1, 0): "  & b) ! Second argument"})

    def test_technical_tokens_inside_the_comment(self):
        source = "! cudaMalloc, nfft, 1024 要素"
        for token, replacement in (("cudaMalloc", "cudaFree"), ("nfft", "nt"), ("1024", "2048")):
            with self.subTest(token=token), self.assertRaises(AuditError):
                self.exercise([[source]], {(0, 0): source.replace("要素", "elements").replace(token, replacement)})

    def test_physical_newline_cannot_extend_a_comment(self):
        source = "! 説明\nn = 1"
        with self.assertRaises(AuditError):
            self.exercise([[source]], {(0, 0): "! Note\nn = 2"})

    def test_untranslated_japanese_in_same_multiline_run(self):
        source = "! 説明\n! 未訳"
        with self.assertRaisesRegex(AuditError, "Japanese"):
            self.exercise([[source]], {(0, 0): "! Note\n! 未訳"})

    def test_remaining_japanese_must_be_in_a_preserved_literal(self):
        with self.assertRaisesRegex(AuditError, "Japanese"):
            self.exercise([["! 説明"]], {(0, 0): "! Some 説明"})

    def test_fortran_concatenation_is_executable_code(self):
        source = "s = 'a!' // 'b' ! 説明"
        self.exercise([[source]], {(0, 0): source.replace("説明", "Note")})
        with self.assertRaises(AuditError):
            self.exercise([[source]], {(0, 0): "s = 'a!' // 'c' ! Note"})

    def test_cpp_comments_and_unary_not(self):
        for source, target in (("if (!ready) f(); // 説明", "if (!ready) f(); // Note"),
                               ("f(); /* 説明 */", "f(); /* Note */")):
            self.exercise([[source]], {(0, 0): target}, role="code_comment")
        with self.assertRaises(AuditError):
            self.exercise([["if (!ready) f(); // 説明"]], {(0, 0): "if (ready) f(); // Note"}, role="code_comment")

    @unittest.skipUnless(os.environ.get("DOTNET_BIN") and os.environ.get("VALIDATOR_DLL"),
                         "Set DOTNET_BIN and VALIDATOR_DLL to run the official Open XML validation")
    def test_openxml_and_preserved_japanese_literal(self):
        self.exercise([['print *, "日本語!"  ! 説明'], ["! FFT", "長"]],
                      {(0, 0): 'print *, "日本語!"  ! Description', (1, 1): " length"}, sdk=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
