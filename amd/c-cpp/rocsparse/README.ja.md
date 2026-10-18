# rocSPARSE C/C++ 教材・benchmark

[English](README.md) | [日本語](README.ja.md)

CPU・HIP・OpenMP target の教材とbenchmarkを実装しました。教材は各1ソースで完結します。
CPU（AOCL Sparse）・HIP・OpenMP targetのSirius実機buildと基本数値smokeを確認済みです。
教材3本と数値検証96件がすべて成功し、誤差は0でした。
[検証したsource snapshotと保存ログ](../../../jobs/sirius/README.md#focused-numerical-smoke)を参照してください。
後続のソース変更、性能、CPU実効worker数はこの結果の検証範囲外です。

FP64の5点Poisson行列によるCSR SpMVを扱います。教材は1024×1024格子、対角4・隣接要素−1、alpha=beta=1、x/yはすべて1です。GPUはgeneric rocsparse_spmvを使用し、workspace照会→確保→preprocess→computeの順に実行します。
CPU productionは既定のoneMKL Sparse、または明示選択のAOCL Sparseです。
AOCLには`GPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE`を指定し、結果は`cpu-aocl-sparse`と記録します。
REFERENCEを明示した場合だけcpu-reference-csrを構築します。依存不足で他backendへ切り替えません。

AOCLではLP64のdescriptor／CSR生成、hint（固定1回）／optimize、SpMV、破棄を教材内で直接呼びます。
CMakeへinclude directory、primary library、順序付き追加libraryを明示できます。
手動教材buildでは`GPU_SUITE_USE_AOCL_SPARSE=1`を定義し、選択した入力をlinkします。
具体例は[英語版](README.md)を参照してください。benchmarkは実API版とAOCL／OMPの設定値を記録し、
実workers数は未観測としてnullを維持します。指定libraryのhashは実ロードDSOの同一性を保証しないため、
今回の数値smokeでは完全な実ロードDSOの同一性や並列worker動作を確認済みとはしません。

computeではwarm-up後と各trial開始前にC/yを復元し、repeat回の更新を連続測定します。
end-to-endではrepeatごとに計測外で復元し、context・descriptorの準備、allocationまたは
OpenMP mapping、転送、計算、完了、結果取得を測定します。明示cleanupと検証は計測外です。

OpenMP側が配列のdevice memoryを管理し、use_device_ptrでrocSPARSEへ渡します。
host fallbackとHIP deviceに対応しないmappingを拒否します。実機では
OMP_TARGET_OFFLOAD=MANDATORYを指定します。CPU/HIP/OpenMPは別build treeを使用し、
native CMake HIPには対応するClangを指定します。Siriusで負荷のあるbuild・test・GPU実行は
承認されたscheduler jobで実施します。

ファイル一覧と一般的なbuild条件は[英語版](README.md)、仕様の正本は
[PROJECT_SPECIFICATION](../../../docs/PROJECT_SPECIFICATION.md)、
[BENCHMARK_PROTOCOL](../../../docs/BENCHMARK_PROTOCOL.md)、
[RESULT_SCHEMA](../../../docs/RESULT_SCHEMA.md)です。
