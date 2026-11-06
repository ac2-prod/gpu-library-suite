# 編集できない図の実装上の注意

図の再作成・ラベル置換を行う場合に参照する。以下は実装方法を選ぶための注意であり、単一のXML構成や解像度を必須にしない。

## 元情報と計測

- 編集可能な原図や、文字・座標・色・線幅等の元情報を取得できる場合は、それを優先する。画像からの計測で置き換えない。
- 描画から計測する場合は、細線、破線、矢じり、ラベルを判別できる解像度を選ぶ。200dpiに固定せず、元画像の解像度と図の細かさに合わせる。
- 計測用の切り出しでは、切り出し位置と縮尺を記録し、画像座標をスライド上の座標へ対応付ける。アンチエイリアス等による計測誤差があるため、スクリプトの測定値だけで再現性を判定しない。

## 図形・グループ・矢印

- グループに置換する場合は、外側の位置・大きさだけでなく、子座標系、親グループによる変換、回転・反転、元画像のトリミング、重なり順を考慮し、元の表示領域を再現する。
- DrawingMLでは `off/ext` と `chOff/chExt` の対応を明示する。同じ `off/ext` を設定したことだけで、倍率1や内部配置の一致を保証しない。
- 図形の形状に応じて `prstGeom`、`custGeom` 等を選ぶ。線端の矢じりを `headEnd/tailEnd` で表現する場合は、線の始点・終点に対する向きと矢じりの形状・大きさを確認する。塗りつぶしの矢印等を一律に線端の矢じりへ置き換えない。
- 接続が必要な矢印はコネクタとして作成し、接続先の図形ID・接続点を設定・確認する。矢じりの指定だけで図形への接続が作られたと判断しない。

## ラベル

- `wrap="none"` や `spAutoFit` を一律に指定しない。`spAutoFit` は文字に合わせて図形自体の大きさを変えるため、原図の領域や関連要素との間隔を崩さないか確認する。`normAutofit` を用いる場合は、自動調整後の描画サイズも確認する。
- ラベルの中心座標が原図と同じでも、訳文の長さによって文字の端と矢印との間隔が変わる。[Skill本体の余白と対応関係の基準](../SKILL.md#書式と部分編集)を適用し、中央・左・右揃えの選択も含め、描画結果でひとまとまりに読めることと正しい対応先を確認する。この基準は再作成した図だけでなく、通常の翻訳対象にも適用する。
- 元の図を保持してラベルを置換する場合は、元の日本語が残らず、背景・線・矢印・隣接要素が欠けないことを確認する。

## パッケージの整合性

- 削除対象のrelationship IDが、所属するXML内の他の要素から参照されていないことを確認する。
- メディアファイルは、スライド・マスター・レイアウト等を含むパッケージ全体からの参照がなくなった場合に限り削除する。共有画像を、図の置換だけを理由に削除しない。
- 必要に応じてコンテンツタイプ定義も整合させ、重複するシェイプIDや参照切れを確認する。生成したPPTXをPowerPointで開き、修復要求がないことと描画を確認する。

## 仕様の参照先

- [グループの座標変換](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.drawing.transformgroup?view=openxml-3.0.1)
- [図形を文字に合わせる自動調整（spAutoFit）](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.drawing.shapeautofit?view=openxml-3.0.1)
- [文字を枠内に合わせる自動調整（normAutofit）](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.drawing.normalautofit?view=openxml-3.0.1)
- [コネクタの接続情報](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.drawing.startconnection?view=openxml-3.0.1)
- [画像への参照](https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.drawing.blip?view=openxml-3.0.1)
