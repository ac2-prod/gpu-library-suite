# rocSOLVER：FP64密行列の連立一次方程式

CPU（LAPACKE）、HIP、OpenMP targetで同じ列優先の `A X = B` を解きます。
対角は `n + 1`、非対角は `1`、右辺は `2n`、厳密解は全成分 `1` です。
教材の既定値は `n=1024`、`nrhs=16`。教材3本はそれぞれ独立した単一ソースです。
ファイル一覧とビルド条件は [README.md](README.md) を参照してください。

CPU教材は `LAPACKE_dgesv`、benchmarkはLU分解と求解を分けます。
GPUではrocBLAS handleを使って `rocsolver_dgetrf` のinfoを確認した後に
`rocsolver_dgetrs` を呼びます。getrsにはinfo出力引数がありません。
記録上の `getrs_info=0` はAPIと完了同期の成功を意味し、未完了時はnullです。

benchmarkは破壊的な分解に対して `repeat=1` とし、trialsで反復します。
computeではhandle・明示workspace・device配列を保持し、分解、info確認、求解、
完了同期を測定します。end-to-endでは設定と転送も含め、入力生成・検証・明示解放は
測定外です。warm-upと各trialの前に配列・pivot・infoを復元します。
OpenMP所有のmappingをHIPで解放せず、実offloadとHIP deviceの対応を確認します。

ローカルの合成APIテストとは別に、承認されたSiriusジョブでCPU・HIP・OpenMP targetの
build、教材3本、数値smoke全96件の成功を確認済みです。
[検証したsource snapshotと保存ログ](../../../jobs/sirius/README.md#focused-numerical-smoke)を参照してください。
性能、CPU実効worker数、後続のソース変更は未検証です。
