# rocBLAS C/C++ 教材・benchmark

[English](README.md) | [日本語](README.ja.md)

CPU・HIP・OpenMP target の教材とbenchmarkを実装しました。教材は各1ソースで完結します。
ローカル検証はhost上のAPI fixtureと数値・失敗処理の検査です。Siriusでは
`sirius-amd-build-20261017T104351Z-a3697cae`で6成果物のbuildが成功し、
保存ログでCPU・HIP教材の数値誤差0も確認しています。OpenMP targetとCPUの
1／2スレッド要求を含むBLAS限定96件のsmokeは、別途承認されたジョブで確認します。
CPU実効スレッド数は未確認であり、要求値を実測値として扱いません。
rocFFTの実機成功はこのライブラリの実機検証を代替しません。

FP64列優先DGEMMを扱います。教材はm=n=k=1024、alpha=beta=1、A/B/Cをすべて1とし、期待値は1025です。CPUは明示したCBLAS provider、GPUはrocblas_dgemmを使用します。

computeではwarm-up後と各trial開始前にC/yを復元し、repeat回の更新を連続測定します。
end-to-endではrepeatごとに計測外で復元し、context・descriptorの準備、allocationまたは
OpenMP mapping、転送、計算、完了、結果取得を測定します。明示cleanupと検証は計測外です。

OpenMP側が配列のdevice memoryを管理し、use_device_ptrでrocBLASへ渡します。
host fallbackとHIP deviceに対応しないmappingを拒否します。実機では
OMP_TARGET_OFFLOAD=MANDATORYを指定します。CPU/HIP/OpenMPは別build treeを使用し、
native CMake HIPには対応するClangを指定します。Siriusで負荷のあるbuild・test・GPU実行は
承認されたscheduler jobで実施します。

ファイル一覧と一般的なbuild条件は[英語版](README.md)、仕様の正本は
[PROJECT_SPECIFICATION](../../../docs/PROJECT_SPECIFICATION.md)、
[BENCHMARK_PROTOCOL](../../../docs/BENCHMARK_PROTOCOL.md)、
[RESULT_SCHEMA](../../../docs/RESULT_SCHEMA.md)です。
