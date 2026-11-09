"""Regression tests for explicitly scoped approvals; only synthetic presentations are created."""
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

import exact_replacements as exact
from apply_translation import apply, expected
from check_changes import masked_equal
from office_package import AuditError, ET, Package, parse_xml, q, xp
from pptx_audit import compare, inventory, template


def digest(data):
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def span(old, new, text):
    return dict(source_start=old.index(text), target_start=new.index(text), text=text)


def quotation(old, new):
    return dict(source_start=old.index("「"), source_end=old.index("」")+1,
                target_start=new.index('"'), target_end=new.rindex('"')+1)


class ApprovalContextTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name).resolve()
        self.counter = 0

    def case(self, original, intended, role, *, spans=None, pairs=None, format_at=None, mono=False):
        self.counter += 1
        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1))
        for text in original:
            run = box.text_frame.paragraphs[0].add_run()
            run.text = text
            run.font.name = "Consolas" if mono else "Yu Gothic"
            run.font.size = Pt(17)
            run.font.bold = True
            run.font.italic = True
            run.font.underline = True
            run.font.color.rgb = RGBColor(23, 84, 127)
            rp = run._r.get_or_add_rPr()
            for attr, value in {"lang":"ja-JP", "kern":"1200", "spc":"15", "baseline":"1000"}.items():
                rp.set(attr, value)
            ea = ET.SubElement(rp, q("a:ea"))
            ea.set("typeface", "Yu Gothic")
        source = self.work / f"source-{self.counter}.pptx"
        deck.save(source)
        pkg = Package(source)
        _, allow = inventory(pkg, [1])
        rows = list(allow["ranges"].values())
        self.assertEqual([r["source"] for r in rows], original)
        ref = dict(unit="unit", page=1, part=pkg.slides[0], kind="slide", shape=rows[0]["shape"]["id"],
                   paragraph=rows[0]["paragraph"], range_ids=[r["id"] for r in rows],
                   source="".join(original), translation="".join(intended), role=role, protected_spans=spans or [])
        if pairs is not None:
            ref["prose_quotes"] = pairs
        manifest = template(allow)
        manifest["translations"] = []
        rules = []
        for i, (row, text) in enumerate(zip(rows, intended)):
            font_only = format_at is not None and i in format_at
            if text == row["source"] and not font_only:
                continue
            changes = format_at[i] if font_only else {"lang":"en-US"}
            rule_role = "format_only" if font_only else role
            fmt = "format_only" if font_only else "preserve_monospace" if mono else "standard_translation"
            entry = dict(**row, target=text, format_changes=dict(changes), reason="Synthetic fixture", keep_reason="")
            manifest["translations"].append(entry)
            rule = dict(**exact.location(row), unit="unit", source_text=row["source"], target_text=text,
                        format_policy=fmt, source_role=rule_role, reason="Synthetic fixture")
            if font_only:
                before = {"latin":row["format_before"]["font"], "lang":"ja-JP"}
                rule.update(expected_before={k:before.get(k) for k in changes}, target_format=dict(changes))
            rules.append(rule)
        return pkg, allow, manifest, rules, [ref]

    def inputs(self, case):
        pkg, allow, manifest, rules, refs = case
        doc = dict(version=1, slides=[1], source_sha256=pkg.sha, manuscript_sha256=digest(refs),
                   approved_exact_replacements=rules)
        manifest["exact_approvals_sha256"] = digest(doc)
        return doc, digest(doc), refs, digest(refs)

    def check(self, case):
        pkg, allow, manifest = case[:3]
        return expected(pkg, allow, manifest, exact_inputs=self.inputs(case))[1]

    def test_registered_prose_quotes_on_english_word(self):
        old, new = "「Introduction」", '"Introduction"'
        self.check(self.case([old], [new], "prose", pairs=[quotation(old, new)]))

    def test_split_prose_quotes_and_technical_anchor(self):
        a, b = ["「教材", "-GPU", "概要」の"], ['See "Guide ', "-GPU", ' Overview"']
        self.check(self.case(a, b, "prose", spans=[span("".join(a), "".join(b), "-GPU")],
                             pairs=[quotation("".join(a), "".join(b))]))

    def test_addressed_ascii_quote_in_japanese_prose(self):
        a, b = '本文では "概要" と呼ぶ', 'The text calls it "Overview".'
        pairs = [dict(source_start=a.index('"'),source_end=a.rindex('"')+1,
                      target_start=b.index('"'),target_end=b.rindex('"')+1)]
        self.check(self.case([a],[b],"prose",pairs=pairs))

    def test_ambiguous_ascii_literal_and_quoted_path_are_protected(self):
        for a,b in [('"結果"','"Result"'),('実行時の "$HOME/file" を参照','See "another-file"'),
                    ('保存先 "/tmp/資料 A" を使う','Use "/tmp/File A"')]:
            c=self.case([a],[b],"prose",pairs=[dict(source_start=a.index('"'),source_end=a.rindex('"')+1,
                                                   target_start=b.index('"'),target_end=b.rindex('"')+1)])
            with self.subTest(source=a),self.assertRaises(AuditError):
                self.check(c)

    def test_two_quote_pairs_with_unicode_offsets(self):
        a, b = "😀「解」 と 「説」", '😀"One" and "Two"'
        pairs = [dict(source_start=1, source_end=4, target_start=1, target_end=6),
                 dict(source_start=a.rindex("「"), source_end=len(a), target_start=b.index('"Two'), target_end=len(b))]
        self.check(self.case([a], [b], "prose", pairs=pairs))

    def test_wrong_quote_positions(self):
        a, b = "「教材」", '"Guide"'
        pair = quotation(a, b)
        for key in pair:
            broken = dict(pair); broken[key] += 1
            with self.subTest(key=key), self.assertRaises(AuditError):
                self.check(self.case([a], [b], "prose", pairs=[broken]))

    def test_missing_quote_pair_or_bound_run(self):
        a, b = ["「教材", "の概要」"], ['"Guide ', 'Overview"']
        for no_rule in (False, True):
            case = self.case(a, b, "prose", pairs=[quotation("".join(a), "".join(b))] if no_rule else [])
            if no_rule:
                case[3].pop()
            with self.subTest(no_rule=no_rule), self.assertRaises(AuditError):
                self.check(case)

    def test_unregistered_ordinary_edit_cannot_introduce_quotes(self):
        case = self.case(["「説明」"], ['"Note"'], "prose")
        with self.assertRaises(AuditError):
            expected(case[0], case[1], case[2])

    def test_lone_quote_cannot_be_added(self):
        a, b = "「説明」", '"Note""'
        with self.assertRaises(AuditError):
            self.check(self.case([a], [b], "prose", pairs=[dict(source_start=0, source_end=len(a),target_start=0,target_end=6)]))

    def test_existing_literal_cannot_be_changed_as_prose(self):
        old = '参照 "result" と 「説明」'
        new = 'See "answer" and "note"'
        pair = dict(source_start=old.index("「"), source_end=len(old), target_start=new.index('"note'), target_end=len(new))
        with self.assertRaises(AuditError):
            self.check(self.case([old], [new], "prose", pairs=[pair]))

    def test_quotes_inside_program_string_cannot_be_relabelled(self):
        a, b = 'puts("「Intro」");', 'puts(""Intro"");'
        pair = dict(source_start=a.index("「"), source_end=a.index("」")+1, target_start=b.index('""')+1, target_end=b.rindex('""')+1)
        with self.assertRaises(AuditError):
            self.check(self.case([a], [b], "prose", pairs=[pair]))

    def test_print_string_bytes(self):
        with self.assertRaises(AuditError):
            self.check(self.case(['print *, "result = "'], ['print *, "result="'], "code"))

    def test_bind_c_and_extern_c(self):
        for a,b in [('bind(C, name="thrust_square_sum")','bind(C, name="other")'),('extern "C"', 'extern "C++"')]:
            with self.subTest(s=a), self.assertRaises(AuditError):
                self.check(self.case([a], [b], "code"))

    def test_inline_print_bind_and_extern_are_not_prose_quotations(self):
        for prefix, token in [('print *, ', "result = "), ('bind(C, name=', "thrust_square_sum"), ('extern ', "C")]:
            a = '説明 ' + prefix + '"' + token + '"'
            b = 'Example: ' + prefix + '"' + token + ' value"'
            pairs = [dict(source_start=a.index('"'), source_end=len(a),
                          target_start=b.index('"'), target_end=len(b))]
            with self.subTest(source=a), self.assertRaises(AuditError):
                self.check(self.case([a], [b], "prose", pairs=pairs))

    def test_explicit_code_context_keeps_the_comment_only_rules(self):
        for old,new,role in [("call f() ! 説明","call f() ! Note","fortran_comment"),
                             ("f(); // 説明","f(); // Note","code_comment")]:
            with self.subTest(role=role):
                c=self.case([old],[new],role,mono=True)
                c[4][0]["role"]="code"
                self.check(c)

    def test_quoted_command_path_and_option_value(self):
        for a,b,role in [('$ nvfortran "a b.f90"','$ nvfortran "b a.f90"',"command"),
                          ('--path=「資料」','--path="data"',"prose"),
                          ('参照「file.f90」','See "file.f90"',"prose")]:
            with self.subTest(s=a), self.assertRaises(AuditError):
                self.check(self.case([a], [b], role, pairs=[quotation(a,b)] if "「" in a else None))

    def test_option_description(self):
        a, b = "-acc：OpenACC directives を有効化", "-acc: Enable OpenACC directives"
        self.check(self.case([a], [b], "option_description", spans=[span(a,b,"-acc")]))

    def test_every_required_option(self):
        for opt in ["-acc","-cuda","-gpu=cc90","-cudalib=cufft","-cudalib=cublas","-cudalib=cusparse",
                    "-cudalib=cusolver","-cudalib=curand","-std=c++17","-c","-c++libs","--mode=dev:debug"]:
            a, b = opt+"：説明", opt+": Description"
            with self.subTest(option=opt):
                self.check(self.case([opt,"：","説明"], [opt,": ","Description"], "option_description", spans=[span(a,b,opt)]))

    def test_token_itself_split_across_runs(self):
        self.check(self.case(["-","gpu","=cc90","：","説明"],["-","gpu","=cc90",": ","Generate code"],
                             "option_description",spans=[dict(source_start=0,target_start=0,text="-gpu=cc90")]))

    def test_changed_flag_or_value(self):
        for a,b in [("-acc","-cuda"),("-gpu=cc90","-gpu=cc80"),("-cudalib=cufft","-cudalib=cublas")]:
            with self.subTest(s=a), self.assertRaises(AuditError):
                self.check(self.case([a+"：説明"], [b+": Note"], "option_description",
                                     spans=[dict(source_start=0,target_start=0,text=a)]))

    def test_option_span_must_be_complete_and_at_start(self):
        a, b = "-gpu=cc90：説明", "-gpu=cc90: Note"
        for spec in ([],[dict(source_start=0,target_start=0,text="-gpu")],[dict(source_start=1,target_start=1,text="gpu=cc90")]):
            with self.subTest(spans=spec),self.assertRaises(AuditError):
                self.check(self.case([a],[b],"option_description",spans=spec))

    def test_colon_in_option_value_is_not_a_separator(self):
        with self.assertRaises(AuditError):
            self.check(self.case(["--gpu=cc90:fast：説明"],["--gpu=cc90: Note"],"option_description",
                                 spans=[dict(source_start=0,target_start=0,text="--gpu=cc90")]))

    def test_option_cannot_gain_suffix_or_move_between_runs(self):
        for src,dst in [(["-acc","：説明"],["-accX",": Note"]),(["-a","cc：説明"],["-ac","c: Note"])]:
            with self.subTest(source=src),self.assertRaises(AuditError):
                self.check(self.case(src,dst,"option_description",spans=[dict(source_start=0,target_start=0,text="-acc")]))

    def test_unregistered_sibling_and_wrong_option_role(self):
        c=self.case(["-acc","：","説明"],["-acc",": ","Description"],"option_description",
                    spans=[dict(source_start=0,target_start=0,text="-acc")])
        c[3].pop()
        with self.assertRaises(AuditError):
            self.check(c)
        c=self.case(["-acc：説明"],["-acc: Description"],"prose")
        with self.assertRaises(AuditError):
            self.check(c)

    def test_shell_command_cannot_claim_option_description(self):
        a,b="$ nvfortran -acc -cuda ! 説明","$ nvfortran -acc -cuda ! Note"
        with self.assertRaises(AuditError):
            self.check(self.case([a],[b],"option_description",spans=[span(a,b,"-acc")]))

    def test_format_only_each_attribute_or_both(self):
        for attrs in ({"latin":"Segoe UI"},{"lang":"en-US"},{"latin":"Segoe UI","lang":"en-US"}):
            with self.subTest(attributes=attrs):
                c=self.case(["NVIDIA GPU, Fortran"],["NVIDIA GPU, Fortran"],"format_only",format_at={0:attrs})
                tree=parse_xml(self.check(c)[c[0].slides[0]])
                rec=c[3][0]; n,=xp(tree,rec["xpath"]); p=n.getparent().find(q("a:rPr"))
                self.assertEqual(n.text,"NVIDIA GPU, Fortran")
                self.assertEqual(p.find(q("a:latin")).get("typeface"),attrs.get("latin","Yu Gothic"))
                self.assertEqual(p.get("lang"),attrs.get("lang","ja-JP"))
                self.assertEqual(p.find(q("a:ea")).get("typeface"),"Yu Gothic")
                masked_equal(c[0].data[c[0].slides[0]],ET.tostring(tree),c[0].slides[0],c[1],c[2])

    def test_format_only_with_translated_neighbor(self):
        c=self.case(["教材","NVIDIA GPU"],["Materials: ","NVIDIA GPU"],"prose",format_at={1:{"latin":"Segoe UI"}})
        self.check(c)

    def test_option_translation_and_font_from_same_source(self):
        c = self.case(["-flag", "：", "Runtime", "を有効化"], ["-flag", ": Enable ", "Runtime", " support"],
                      "option_description", spans=[dict(source_start=0,target_start=0,text="-flag")],
                      format_at={2:{"latin":"Segoe UI","lang":"en-US"}})
        self.check(c)
        self.assertEqual(xp(c[0].xml[c[0].slides[0]], ".//a:t")[0].text, "-flag")
        c[3].append(deepcopy(c[3][0]))
        out = self.work / "conflict.pptx"
        with self.assertRaises(AuditError):
            apply(c[0].path, out, c[1], c[2], exact_inputs=self.inputs(c))
        self.assertFalse(out.exists())

    def test_prose_punctuation_and_digits_keep_their_text(self):
        for token in (" (", ")", " / ", "12", ","):
            with self.subTest(token=token):
                self.check(self.case(["解説", token],["Example ", token],"prose",format_at={1:{"latin":"Segoe UI"}}))
        for parts in (["x", "/", "y"], ["callfn", "(", "n", ")"], ["\"","12","\""]):
            with self.subTest(parts=parts), self.assertRaises(AuditError):
                self.check(self.case(parts,parts,"format_only",format_at={1:{"latin":"Segoe UI"}}))

    def formatted_case(self, pkg, allow, row, role):
        ref=dict(unit="example",page=1,part=row["part"],kind="slide",shape=row["shape"]["id"],
                 paragraph=row["paragraph"],range_ids=[row["id"]],source=row["source"],
                 translation=row["source"],role=role,protected_spans=[])
        edit=dict(**row,target=row["source"],format_changes={"latin":"Segoe UI"},reason="Source-bound example")
        rule=dict(**exact.location(row),unit="example",source_text=row["source"],target_text=row["source"],
                  source_role="format_only",format_policy="format_only",expected_before={"latin":"Yu Gothic"},
                  target_format=edit["format_changes"],reason=edit["reason"])
        man=template(allow);man["translations"]=[edit]
        return pkg,allow,man,[rule],[ref]

    def test_source_cell_address_and_product_spelling(self):
        deck=Presentation();slide=deck.slides.add_slide(deck.slide_layouts[6])
        table=slide.shapes.add_table(1,2,Inches(1),Inches(1),Inches(4),Inches(1)).table
        for cell in table.rows[0].cells:
            cell.text="hipExample"
            cell.text_frame.paragraphs[0].runs[0].font.name="Yu Gothic"
        path=self.work/"cells.pptx";deck.save(path);pkg=Package(path)
        _,allow=inventory(pkg,[1]);r,s=list(allow["ranges"].values())
        self.assertNotEqual(r["cell"],s["cell"])
        c=self.formatted_case(pkg,allow,r,"format_only")
        part=self.check(c)[r["part"]];masked_equal(pkg.data[r["part"]],part,r["part"],allow,c[2])
        c[3][0]["xpath"]=s["xpath"]
        with self.assertRaises(AuditError):
            self.check(c)

    def test_named_display_range_keeps_technical_spelling(self):
        c=self.case(["対応する","sample_api","を呼ぶ"],["Use ","sample_api"," here."],"prose",
                    format_at={1:{"latin":"Segoe UI"}})
        name=c[1]["ranges"][c[3][1]["range_id"]]["id"]
        c[4][0]["protected_spans"]=[span(c[4][0]["source"], c[4][0]["translation"], "sample_api")]
        c[4][0]["display_font_ranges"]=[name]
        self.check(c)
        c[4][0]["display_font_ranges"]=[name,name]
        with self.assertRaises(AuditError):
            self.check(c)
        code=self.case(["call ","sample_api","()"],["call ","sample_api","()"],"format_only",
                       format_at={1:{"latin":"Segoe UI"}})
        code[4][0]["display_font_ranges"]=[code[3][0]["range_id"]]
        with self.assertRaises(AuditError):
            self.check(code)

    def test_command_continuation_in_original_text_body(self):
        deck=Presentation();slide=deck.slides.add_slide(deck.slide_layouts[6])
        box=slide.shapes.add_textbox(Inches(1), Inches(1), Inches(7), Inches(1))
        box.text_frame.paragraphs[0].text="$ compiler -x \\"
        p=box.text_frame.add_paragraph();p.text="  example"
        p.runs[0].font.name="Yu Gothic";path=self.work/"cmd.pptx";deck.save(path);pkg=Package(path)
        _,allow=inventory(pkg,[1]);r=list(allow["ranges"].values())[-1]
        with self.assertRaisesRegex(AuditError,"continuation"):
            self.check(self.formatted_case(pkg,allow,r,"format_only"))

    def test_unregistered_and_wrong_shape_format(self):
        for wrong in ("unregistered","wrong shape","wrong source","stale font"):
            c=self.case(["English"],["English"],"format_only",format_at={0:{"latin":"Segoe UI"}})
            rule=c[3][0]
            if wrong=="unregistered":c[3].clear()
            elif wrong=="wrong shape":rule["shape_id"]="999"
            elif wrong=="wrong source":rule["source_text"]="Other"
            else:rule["expected_before"]["latin"]="Other"
            with self.subTest(case=wrong),self.assertRaises(AuditError):
                self.check(c)

    def test_format_only_rejects_changed_text_and_nonenglish(self):
        for label,text in [("English","english"),("日本語 English","日本語 English"),("!$omp parallel","!$omp parallel")]:
            with self.subTest(label=label),self.assertRaises(AuditError):
                self.check(self.case([label],[text],"format_only",format_at={0:{"latin":"Segoe UI"}}))

    def test_format_only_rejects_code_monospaced_and_protected_spans(self):
        for text,kind,mono in [("NVIDIA GPU","prose",True),("compute","code",False),
                               ("$ run --target","command",False),("call f()","format_only",False),
                               ("data_value","format_only",False),("extern \"C\"","format_only",False)]:
            with self.subTest(text=text),self.assertRaises(AuditError):
                self.check(self.case([text],[text],kind,format_at={0:{"latin":"Segoe UI"}},mono=mono))
        with self.assertRaises(AuditError):
            self.check(self.case(["value"],["value"],"prose",spans=[span("value","value","value")],
                                 format_at={0:{"latin":"Segoe UI"}}))

    def test_format_only_cannot_change_an_extra_attribute(self):
        for extra in ("size_pt","color","bold","ea","underline","baseline","spc","kern"):
            c=self.case(["English"],["English"],"format_only",format_at={0:{"latin":"Segoe UI"}})
            c[2]["translations"][0]["format_changes"][extra]=1
            with self.subTest(attr=extra),self.assertRaises(AuditError):
                self.check(c)

    def test_format_only_cannot_move_shape(self):
        c=self.case(["English"],["English"],"format_only",format_at={0:{"latin":"Segoe UI"}})
        geom=next(iter(c[1]["geometry"].values()))
        c[2]["adjustments"].append(dict(id=geom["id"],before=geom["source"],after=int(geom["source"])+1000,reason="move"))
        with self.assertRaisesRegex(AuditError,"geometry"):
            self.check(c)

    def test_layout_changes_have_their_own_source_bound_grants(self):
        c=self.case(["英文","Product"],["English ","Product"],"prose",format_at={1:{"latin":"Segoe UI"}})
        r=c[1]["ranges"][c[3][-1]["range_id"]]; part=r["part"]
        c[2]["layout_sizes"]=[dict(id=r["id"],before=17,after=15,reason="Measured overflow at 17 pt")]
        node,=xp(c[0].xml[part],r["xpath"])
        body=node.getparent().getparent().getparent()
        from office_package import address
        bp=body.find(q("a:bodyPr"))
        c[2]["frame_layout"]=[dict(part=part,shape_id=r["shape"]["id"],xpath=address(bp),attribute="rIns",
                                   before=bp.get("rIns"),after=12700,reason="Keep the label clear of adjacent artwork")]
        data=self.check(c)[part];masked_equal(c[0].data[part],data,part,c[1],c[2])
        c[2]["layout_sizes"][0]["before"]=16
        with self.assertRaises(AuditError):
            self.check(c)

    def test_unchanged_command_protects_shared_frame_from_movement(self):
        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        shape = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(7), Inches(1))
        shape.text_frame.paragraphs[0].text = "$ compiler input.o"
        caption = shape.text_frame.add_paragraph()
        caption.text = "Caption"
        caption.runs[0].font.name = "Yu Gothic"
        path = self.work / "command-frame.pptx"
        deck.save(path)
        pkg = Package(path)
        _, allow = inventory(pkg, [1])
        row = next(x for x in allow["ranges"].values() if x["source"] == "Caption")
        case = self.formatted_case(pkg, allow, row, "format_only")
        geom = next(iter(allow["geometry"].values()))
        case[2]["adjustments"].append(dict(id=geom["id"], before=geom["source"],
                                         after=int(geom["source"]) + 12700, purpose="text_layout",
                                         reason="Move the caption"))
        with self.assertRaisesRegex(AuditError, "geometry"):
            self.check(case)

    def test_audit_detects_collateral_change(self):
        c=self.case(["English"],["English"],"format_only",format_at={0:{"latin":"Segoe UI"}})
        part=c[0].slides[0]; tree=parse_xml(self.check(c)[part]); path=c[3][0]["xpath"]
        for attr in ("b","i","sz","spc","kern","baseline","u"):
            other=deepcopy(tree); n,=xp(other,path);n.getparent().find(q("a:rPr")).set(attr,"0")
            with self.subTest(attr=attr),self.assertRaisesRegex(AuditError,"Unlisted"):
                masked_equal(c[0].data[part],ET.tostring(other),part,c[1],c[2])

    def test_audit_preserves_color_east_asian_font_and_position(self):
        c = self.case(["English"], ["English"], "format_only", format_at={0:{"lang":"en-US"}})
        part = c[0].slides[0]
        clean = parse_xml(self.check(c)[part])
        for path, attribute, value in [(".//a:srgbClr", "val", "FF0000"),
                                       (".//a:ea", "typeface", "Other"), (".//p:spPr/a:xfrm/a:off", "x", "1")]:
            with self.subTest(attribute=attribute), self.assertRaisesRegex(AuditError, "Unlisted"):
                other = deepcopy(clean)
                xp(other, path)[0].set(attribute, value)
                masked_equal(c[0].data[part], ET.tostring(other), part, c[1], c[2])

    def test_font_grant_with_wrong_language_or_policy(self):
        for bad in ("en-GB", "other policy"):
            c = self.case(["English"], ["English"], "format_only", format_at={0:{"lang":"en-US"}})
            if bad == "en-GB":
                c[3][0]["target_format"]["lang"] = bad
                c[2]["translations"][0]["format_changes"]["lang"] = bad
            else:
                c[3][0]["format_policy"] = "standard_translation"
            with self.subTest(bad=bad), self.assertRaises(AuditError):
                self.check(c)

    def test_no_implicit_option_separator_from_ordinary_runs(self):
        for old_delim, new_delim, target in [("：", ": ", "Description"), (":", ":", "value")]:
            c = self.case(["-acc", old_delim, " 説明"], ["-acc", new_delim, target], "option_description")
            with self.subTest(delimiter=old_delim), self.assertRaises(AuditError):
                expected(c[0], c[1], c[2])

    @unittest.skipUnless(os.environ.get("DOTNET_BIN") and os.environ.get("VALIDATOR_DLL"), "Provide the official SDK validator")
    def test_official_openxml_and_shared_edit_compare_paths(self):
        mixed = self.case(["構造","最適化"],["the structure ","and the optimization"],"prose")
        mixed[3].pop()
        mixed[4][0].pop("role")
        mixed[4][0].pop("protected_spans")
        for c in [
            mixed,
            self.case(["「Introduction」"],['"Introduction"'],"prose",pairs=[quotation("「Introduction」",'"Introduction"')]),
            self.case(["-acc","：","説明"],["-acc",": ","Description"],"option_description",
                      spans=[dict(source_start=0,target_start=0,text="-acc")]),
            self.case(["-flag","：","Runtime","を利用"],["-flag",": ","Runtime"," support"],"option_description",
                      spans=[dict(source_start=0,target_start=0,text="-flag")],format_at={2:{"latin":"Segoe UI"}}),
            self.case(["NVIDIA GPU, Fortran"],["NVIDIA GPU, Fortran"],"format_only",
                      format_at={0:{"latin":"Segoe UI","lang":"en-US"}})
        ]:
            pkg,allow,man=c[:3]; inputs=self.inputs(c)
            out=self.work/(pkg.path.stem+"-candidate.pptx")
            apply(pkg.path,out,allow,man,exact_inputs=inputs)
            report=compare(pkg.path,out,allow,man,os.environ["DOTNET_BIN"],os.environ["VALIDATOR_DLL"],exact_inputs=inputs)
            self.assertEqual(report["status"],"PASS")
            self.assertEqual(report["source_sdk"]["errors"],[])
            self.assertEqual(report["english_sdk"]["errors"],[])
            self.assertEqual(report["new_sdk_errors"],[])
            self.assertTrue(report["approved_exact_replacements"])


if __name__=="__main__":
    unittest.main(verbosity=2)
