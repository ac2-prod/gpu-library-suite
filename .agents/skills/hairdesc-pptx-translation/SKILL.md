---
name: hairdesc-pptx-translation
description: HAIRDESCの日本語PowerPoint教材を既存の構造と書式を保って英訳するとき、または英訳版を日本語正本と照合するときに使用する。
---

# HAIRDESC教材のPowerPoint英訳

依頼に含まれる教材・ページ・作業段階を最初に記録する。主担当だけがPPTXを編集する。日本語版が正本。教材別・版別に独立したローカル作業領域 `<原文stem>_work/<run>/` を用意する。原本のSHA-256を記録してローカルに複製し、以後はコピーを使う。PPTX、PDF、PNG、メトリクス、監査記録はSSHFS、ネットワーク共有、クラウド同期フォルダの外に置く。

**レイアウトの基準はMicrosoft PowerPoint自身が出力したPDF。** 原文・英訳・枠線版は同じ端末、OS、PowerPointビルド、同じフォント環境で直列に描画する。LibreOfficeをレンダリング・PDF生成・合否判定に使用しない。Computer UseでPowerPointやターミナルを操作しない。PDFからの画像化にはPopplerを使用できる。

## 使い始める

呼び出し例：`$hairdesc-pptx-translation 指定した教材・ページの棚卸しだけを行ってください`。翻訳依頼の場合は原本・ページ範囲・原文保持の例外を明示する。範囲を示す引数は `all` または1始まりのカンマ区切り。プロジェクトのパイロット候補は **HAIRDESC_AMD_GPU_Library_F_20261027.pptx のp.3、5、11、12**（表、模式図、コードと矢印、性能図）。パイロットの英訳には別途その指示が必要。教材名・選択ページはレンダラーに組み込まない。

以下のコマンドはリポジトリ直下から実行する。`QA_PYTHON` は lxml、pypdf、pdfplumber、pypdfium2、Pillow を利用できる同じPython 3.10以降の実行ファイル。検査専用資料の生成には python-pptx 同梱の空のOfficeテンプレートと日本語フォントを使う。実教材は専用エディタで部分更新する。既存環境を優先し、必要なら作業領域内の独立環境を明示的に用意する。Skillはパッケージを自動導入しない。

```sh
s=.agents/skills/hairdesc-pptx-translation
c="$s/scripts/common"
sel=3,5,11,12
: "${QA_WORK:?新しいローカル領域}" "${QA_PYTHON:?Pythonの絶対パス}" "${JAPANESE_PPTX:?正本の絶対パス}"
mkdir -m 700 "$QA_WORK" && mkdir "$QA_WORK/ja" "$QA_WORK/en" "$QA_WORK/boxed"
cp -n "$JAPANESE_PPTX" "$QA_WORK/ja/render_check.pptx"
jp="$QA_WORK/ja/render_check.pptx"
en="$QA_WORK/en/render_check.pptx"
"$QA_PYTHON" -B "$c/pptx_audit.py" inventory --source "$jp" --slides "$sel" --out "$QA_WORK/inventory"
```

既存の作業領域・完成版に上書きしない。上の `mkdir` が失敗したら停止する。`inventory.json` はスライド順、非表示、Notes Slide、シェイプ・グループ・表、Chart/SmartArt、画像、代替テキスト、埋め込み、アニメーション、部品ハッシュと参照元を示す。`allowlist.json` は原本ハッシュ、対象、部品、名前空間付きXPath・元値を束縛する。`manifest.json` は主担当が記入する翻訳マニフェスト。3ファイルとも検査記録として保持する。

## 何を訳すか

- 意味、トーン、文字の強調、色、階層を引き継ぐ。本文、表の項目、図中ラベル、表示フッター、アクセシビリティ用の説明、発表者ノート本文を確認する。用語集の「次世代HPC・AI研究開発支援センター」は **Advanced HPC-AI Research and Development Support Center (HAIRDESC)**。性能の表示用語は **library / end-to-end** とし、コード中の識別子を保持する。
- マニフェストの各範囲には元ページ、部品、シェイプID、グループ・セル、段落、XPath、原文、書式、保護トークンが入る。対応する `target`、変更する書式 `format_changes` と理由を記入する。数値と単位、URL、API・関数、ファイル名・版、コマンド・引数、コードを維持する。必要な専門語は `extra_protected` に追加する。リンク先と、リンク付きの表示文は別に扱う。混在書式・リンク・フィールドをまたいで文字列を結合して書き直さない。
- 全Notes Slideの全テキスト保持シェイプを調べる。訳すのはbodyプレースホルダーの本文だけ。日付・番号等の自動フィールド、ノートのヘッダー・フッター、画像を維持する。本文外にユーザー本文や分類できない表示物があれば停止する。ノート部品のあるページ、非空の本文を持つページ、各本文の段落数（空段落を含む）を前後で照合する。ノートのないページに追加しない。
- **画像は描画で確認する。** 日本語検出とOCRは補助。PNG/JPEG/EMF/WMF/SVGの形式だけで編集可能とは判定しない。ロゴは日本語を含んでも保持する。既に英語の図、保護ロゴ、非表示の継承プレースホルダー等の判断は部品・位置・証拠を `visual_reviews` に記入する。原図の使用、ネイティブ図形での再作成、元画像を保持したラベル置換を検討する際は [編集不能図の規則](references/non-editable-figures.md) を読む。共有画像・共有データ等を変える必要があればその箇所を保留し、別の方法について所有者と合意してから扱う。判読できないラベル・接続先を推測しない。
- 日本語の意図的な本文保持には、翻訳マニフェストとは別に所有者承認済みのJSONを用意する。内容は `source_sha256`、`slides`、`approved`（厳密な範囲ID→理由）。`keep_reason` と同じ理由を記録し、編集・監査の両コマンドに `--exceptions <file>` を渡す。マニフェストの自己申告だけでは本文保持を認めない。

## 書式と部分編集

英文プロポーショナル書体は **Segoe UI**。本文は原文の適合サイズを尊重し、原則12 pt以上の整数。フッターは例外。コード等の等幅書体とサイズを保つ。原文に明示値がなければ、段落・プレースホルダー・マスター・テーマの継承と自動調整を確認し、未確定の描画値を推測で埋めない。

訳文を原文サイズで描画する。干渉する枠の幅、改行、行間、配置の順に検討し、原文で一行ならできる限り一行を維持する。`wrap`・マージン・自動調整を考慮し、自動縮小枠は寸法を保つ。それでも必要なときだけ該当範囲を縮小する。全体を一括縮小しない。サイズの原文指定値・描画値、変更値、理由、元サイズでの描画結果をページ・シェイプ・段落別に残す。幅等を後から変更したら元サイズに戻して再描画し、不要になった縮小を取り消す。

**余白と対応関係は翻訳対象の全ページで確認する。** 日英のPowerPoint製PDFを並べ、文字の長さや改行が変わった箇所では、枠や中心座標だけでなく、実際に見える文字の端と対応する番号・矢印・図形・吹き出しとの間隔を比較する。原文ではひとまとまりに読めた要素が離れて見える、または別の要素に対応するように見える場合は、はみ出しがなくても修正対象とする。

対応関係が明確なら、追加指示を待たず主担当が関係する要素だけを局所調整し、再描画してまとまり・対応先・干渉の有無を確認する。まずラベル側の調整を検討し、原文の矢印の始終点、形状、向き、線種、色、接続先、重なり順を保つ。原文で意図された余白は保持し、空きが増えたことだけを理由に動かさない。対応先を特定できない箇所は、位置と判断できない理由を報告して保留する。

`format_changes` の例は `{"latin":"Segoe UI","lang":"en-US","size_pt":16}`。不要なキーを入れない。局所配置変更は既存の `adjustments` に棚卸し済みgeometry ID、`before`（元XMLの文字列）、`after`（整数EMU）、`reason` を記す。`reason` には対応する要素、移動方向・量（ptまたはmm）、余白を調整する理由を残す。ページ・図形IDは個別記録に用い、特定ページや固定距離を一般の合否条件にしない。許可リストの範囲外の変更は行わず、その制約を報告する。

```sh
a="$QA_WORK/inventory/allowlist.json"
m="$QA_WORK/inventory/manifest.json"
"$QA_PYTHON" -B "$c/apply_translation.py" --source "$jp" --output "$en" --slides "$sel" --allowlist "$a" --manifest "$m"
"$QA_PYTHON" -B "$c/pptx_audit.py" compare --source "$jp" --english "$en" --slides "$sel" --allowlist "$a" --manifest "$m" --dotnet "$DOTNET_BIN" --validator "$VALIDATOR_DLL" --out "$QA_WORK/comparison.json"
"$QA_PYTHON" -B "$c/boxed_copy.py" --source "$en" --output "$QA_WORK/boxed/render_check.pptx" --slides "$sel"
```

エディタは入力から許可リストを再計算し、許可された対象だけを適用する。監査側は対象外部品の展開後バイト列を照合し、対象XMLでは承認した値を元に戻して残りを独立に比較する。マスター、レイアウト、テーマ、画像、埋め込み、relationship、アニメーション、対象外のノート・ページを保護する。未知の構造・共有データ・許可外差分・参照切れを成功扱いにしない。原本と候補に同じ公式 OpenXmlValidator（Microsoft365プロファイル、全エラー）を実行し、種類・ID・部品・位置・発生数を含む差集合の新規0件を要求する。既存エラーは原本の記録に残す。

## PowerPointでの確認

Macでは所有者が指定した `~/Documents/HAIRDESC-PPTX-Work/` 配下のローカルコピーを使う。SSHFS、`/private/tmp`、PowerPointのアプリコンテナを作業領域にしない。入口は `scripts/macos/render.py`。排他ロック下で固定入力 `~/Documents/HAIRDESC-PPTX-Work/exchange/render-input.pptx` へ複製し、`/usr/bin/open` を1回だけ送り、コンパイル済みAppleScriptで名前とfull nameを照合する。照会はAppleEvent 3秒、Shell側最大5秒、全体120秒。openを再送しない。既存のPowerPointに資料があれば停止し、正常時もPowerPoint全体は終了しない。異常時はopen要求・対象同定・PDF書出し開始の状態と所有PIDを保存し、不明な資料やロックを自動整理しない。

固定入力を上書きする前に、PowerPointに資料が開いていないことを照会する。入力元のコピー前後と固定入力のSHA-256を照合する。固定一時PDFは同じexchangeの `render-output.pending.pdf`。前回の未完了状態・既存pending PDF・所有不明の入力があれば停止し、削除して再開しない。PDFの存在・非空・可読性・全ページ数を検査してから最終PDFへ確定し、同定した資料だけを閉じる。固定入力は保持し、実行ID別のログ・入力/PDFハッシュを別に残す。watcherと直接実行を同時に使わない。

アクセス許可の操作は所有者が行う。修復ダイアログでは停止し、修復後の資料を原本の正常描画として採用しない。表示時とPDF出力時の許可を別々に記録し、表示できただけで無人実行可能と判定しない。初回検証は `--observation <JSON>` を指定し、`owner_confirmed: true`、`repair: false` と許可の観察を記録してからPDFへ進む。
```sh
exchange="$HOME/Documents/HAIRDESC-PPTX-Work/exchange"
mkdir -p "$exchange"
for part in ja en boxed; do
  "$QA_PYTHON" -B "$s/scripts/macos/render.py" \
    "$QA_WORK/$part/render_check.pptx" "$QA_WORK/$part/render_check.pdf" \
    --exchange "$exchange" --report "$QA_WORK/$part/render-report.json" || exit 1
done
"$QA_PYTHON" -B "$c/pdf_audit.py" --pptx "$en" --pdf "$QA_WORK/en/render_check.pdf" \
  --original-pdf "$QA_WORK/ja/render_check.pdf" --manifest "$m" --comparison "$QA_WORK/comparison.json" --out "$QA_WORK/pdf-audit.json"
```

非表示の元ページ番号を記録する。現行Mac入口は入力の書体・表示設定を変更しない。非表示ページの包含と特殊fixtureの実機検証は保留し、必要時はPowerPoint自身で作成・保存した既知正常な検査資料で確認する。最終PDFと同じファイルシステムの一時PDFを検査し、正常な場合だけ出力を確定する。ロックは別プロジェクトとも共通。競合、権限・ファイルアクセスダイアログ、PowerPoint未導入、人的編集中、所有不明、ページ欠落・破損で停止。所有不明時は一時ファイル・ロックを残し、利用者の資料・PowerPointを閉じない。中断ロックの整理は所有者が資料の状態とロック記録を確認してから行う。

独立の検査資料で、PowerPoint内でSegoe UIを指定したRegular/Boldの英文がPDFでどの埋め込みフォントを使ったか調べる。Shellのフォント一覧にないことだけでは停止しない。実英訳でも各翻訳範囲を英訳PDFのページ・文字・位置に照合し、元指定・継承、実フォント、PDF上のサイズと未確認を区別する。ノートはOOXML書式検査、代替テキストは非表示。macOSの収まり判定は**PowerPoint製PDFと枠線付き版の目視**であり、PowerPointのテキスト枠実測メトリクスではない。置換、アウトライン化、重複文字列などで対応を特定できない場合は保留する。

原文・英訳・枠線版を同じ解像度でページ順に比較する。標準は **300 dpi**（例 `pdftoppm -f 5 -l 5 -r 300 -png "$QA_WORK/en/render_check.pdf" "$QA_WORK/en/p5"`）。細線・矢じり・文字の衝突を判別しにくければ、同じ箇所だけ600 dpiで確認する。英訳による文字形状の差があるので、単純な画素一致率を合否基準にしない。

WindowsはWindows PowerShellからの直接実行が標準。作業先・入力はローカルディスクに置く。

```powershell
$s = Join-Path $PWD '.agents\skills\hairdesc-pptx-translation\scripts\windows'
& "$s\render.ps1" -Source "$env:QA_WORK\en\render_check.pptx" -Output "$env:QA_WORK\en\render_check.pdf" -Python $env:QA_PYTHON
# 監視を選ぶ場合だけ別の実行で起動する。直接実行とは排他。
& "$s\render_watch.ps1" -Root $env:QA_WORK -Python $env:QA_PYTHON
```

Windows版のメトリクスには表セル・入れ子グループの文字領域を含める。refitは明示的な `-Refit` と別の `.refit.pdf` 出力を指定した診断用コピーだけ。本番の確認は通常版で行う。Windows COMが実機未試験の段階ではその旨を明記する。

## 再現試験と引渡し

公式.NET 10 SDKはローカルの `DOTNET_BIN` を使用する。検証器のソースをローカル作業先へ複製し、同梱の `packages.lock.json` を使う。初回の承認済み取得に限り次のrestoreを行い、SDKの版・配布元のSHA-512とNuGetのcontentHashを記録する。

```sh
export DOTNET_CLI_HOME="$QA_WORK/dotnet-home" DOTNET_CLI_TELEMETRY_OPTOUT=1
export NUGET_PACKAGES="$QA_WORK/packages"
cp -R "$c/openxml-validator" "$QA_WORK/validator"
"$DOTNET_BIN" restore "$QA_WORK/validator/Validator.csproj" --locked-mode --configfile "$QA_WORK/validator/NuGet.Config" -p:NuGetAudit=false
"$DOTNET_BIN" build "$QA_WORK/validator/Validator.csproj" -c Release --no-restore
VALIDATOR_DLL="$QA_WORK/validator/bin/Release/net10.0/Validator.dll"
"$QA_PYTHON" -B "$c/verify_fixture.py" --platform static --dotnet "$DOTNET_BIN" --work "$QA_WORK/static"
# 合成fixtureのmacOS描画は停止中。既知正常な資料には上のrender.pyを使用する。
```

静的監査用fixtureは原本とは独立した2枚（うち1枚非表示）。PowerPoint互換性は未確認であり、実描画検証の入力としては使用しない。画像内日本語、接続済みコネクタ、自動縮小枠、箇条書き、混在書式、ハイパーリンク、ノート本文・番号を含む。自動検査結果と人による視覚判定を分けて記録する。

レビュー担当はすべて読み取り専用。使えない場合は主担当が次の5工程を独立に順番に行う。対象ハッシュ、ページ・シェイプID、指摘、証拠、判定、担当方式を作業記録に残す。

1. **翻訳内容確認** — 意味、トーン、用語、数値、コード、表・画像内・代替テキスト・ノートの訳抜けと不必要な追記。
2. **ページレイアウト確認** — 原文と通常英訳のPDFを並べ、強調、色、行間、表、図・本文の重なりと、上記の余白基準による要素のまとまりを全対象ページで確認。意図された余白と、翻訳によって生じた不自然な離れを区別する。
3. **矢印・接続確認** — 始終点の相手、端の形状・向き・線種を確認し、番号・ラベルの見える文字端と対応先との間隔を日英で比較する。離れすぎや別の対応先への誤読がないかを判定し、局所調整後も元の対応関係を保持する。
4. **テキストボックス確認** — 枠線版でwrap、内側余白、セル境界、縮小の必要性、はみ出しを確認。
5. **最終チェックリスト確認** — 原本ハッシュ、対象部品・ページとノート数、保護トークン、新規SDKエラー0、実フォント、最終版の通常PDF、1–4の未解決項目を突き合わせる。

指摘は主担当が修正し、影響する監査とレビューを新しいハッシュでやり直す。既存のパッケージ監査は許可した変更だけかを、PDF監査は文字・書体等を検証するもので、意味上のまとまりの合否を代替しない。変更していない部分は確定済みPDF・画像・監査結果を再利用し、追加描画・検査は影響範囲に絞る。全編の依頼で全工程が完了した場合だけ、ハッシュを確認した英訳PPTXを正本の隣へ `<原文stem>_EN.pptx` として排他的に新規コピーし、反映後のハッシュを照合する。パイロット、検査用の枠線版・refit版、未検証の部分訳はローカル作業領域に保持する。
