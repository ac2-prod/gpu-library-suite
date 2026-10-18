# rocRAND C/C++

CPU・HIP・OpenMP targetのSirius実機build、教材3本、数値smoke全96件の成功を確認済みです。
[検証したsource snapshotと保存ログ](../../../jobs/sirius/README.md#focused-numerical-smoke)を参照してください。
確認したCPU backendは逐次実行です。性能と後続のソース変更は未検証です。

教材は単一ソースです。既定問題はFP64一様乱数`1 << 24`個、seed 1234、offset 0。
CPUは`std::mt19937_64`と区間`[0,1)`、GPUは`ROCRAND_RNG_PSEUDO_DEFAULT`、
default orderingと区間`(0,1]`です。同じ分布・出力型の仕事を比較するものであり、
乱数アルゴリズムや各要素の一致は要求しません。

benchmarkのcompute scopeはgenerator・bufferを保持し、warm-up後とtrial前に
seed・offsetを復元して明示的に初期化します。初期化を除いた生成と完了待ちを
測ります。end-to-endは各repeatで作成・初期化・生成・回収までを測り、明示的な
破棄・解放は計測後です。範囲・平均・0.5を中心とする二次モーメントを検証します。

OpenMP版は実offloadとmappingのHIP deviceを確認し、host fallbackを拒否します。
OpenMPのmappingはOpenMPが所有し、rocRANDはそのdevice pointerだけを利用します。

ファイル一覧・手動buildの説明は[英語版](README.md)、規範は
[仕様書](../../../docs/PROJECT_SPECIFICATION.md)と
[benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md)を参照してください。
ローカルfake検証は実GPU検証とは区別します。
