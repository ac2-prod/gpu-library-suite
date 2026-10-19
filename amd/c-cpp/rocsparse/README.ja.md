# rocSPARSE C/C++ 教材・benchmark

[English](README.md) | [日本語](README.ja.md)

CPU・HIP・OpenMP target の教材とbenchmarkを実装しました。教材は各1ソースで完結します。
v2/CSR rowsplit版`a974467bd4f9b45526a0c448f597f895cd244062`について、
CPU（AOCL Sparse）・HIP・OpenMP targetの6成果物をSiriusでbuildし、
教材3本（最大誤差0）と数値smoke 96件がすべて成功しました。
別campaign `sirius-rocsparse-v2-20261019T010457Z-a974467`では、
各wave 6物理ノード・2 waveの本測定1,080行がすべて数値検証PASSです。
raw・来歴・再生成した性能図は同campaignの`runs/`配下に非配布の証跡として保持します。
[以前の基本smoke](../../../jobs/sirius/README.md#focused-numerical-smoke)とadaptive結果は、別の履歴として保持します。
CPU実効worker数と完全な実ロードDSOの同一性は引き続き未確認です。
後続のソース変更は、この結果の検証範囲外です。

FP64の5点Poisson行列によるCSR SpMVを扱います。教材は1024×1024格子、対角4・隣接要素−1、alpha=beta=1、x/yはすべて1です。GPUは`rocsparse_v2_spmv`と`rocsparse_spmv_alg_csr_rowsplit`を固定使用します。
SpMV descriptorに非転置・FP64 scalar/compute型を設定し、analysisとcomputeのworkspaceサイズを個別に照会します。
analysisを1回実行してそのbufferを解放した後、compute用bufferを確保します。
library（内部scope名`compute`）では準備済みdescriptorとcompute bufferを再利用し、end-to-endでは準備も計測に含めます。
v2 APIが利用できなければ対象GPU targetを依存不足として無効化し、旧APIへfallbackしません。
旧adaptive結果は履歴として保持し、新しいv2/rowsplit campaignへ混在させません。
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
