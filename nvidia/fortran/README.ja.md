# NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

6ライブラリそれぞれにCPU、直接CUDA Fortran、OpenACCの教材サンプルと
ベンチマークを実装しています。**ソース追加済みであり、NVHPC/GPUでの実行確認済みではありません。**
入力は2026-09-17版Fortran教材の掲載コード抽出一式です。
既存C/C++計算本体、過去のflags・測定値、承認済み図は変更していません。
保存済み測定データ、実機ログ、配布候補archiveは同梱しません。

<a id="source-map"></a>

## コード対応

| ライブラリ | CPU / 直接CUDA / OpenACCのstem | CPU計算 | 教材の問題 |
| --- | --- | --- | --- |
| [cuFFT](cufft/README.ja.md) | `fft_cpu` / `fft_gpu` / `openacc_cufft` | FFTW Fortran interface、逐次 | FP32複素C2C、長さ1024、batch 4096 |
| [cuBLAS](cublas/README.ja.md) | `blas_cpu` / `blas_gpu` / `openacc_cublas` | Fortran DGEMM、oneMKL | FP64、M=N=K=1024、alpha=beta=1 |
| [cuSPARSE](cusparse/README.ja.md) | `sparse_cpu` / `sparse_gpu` / `openacc_cusparse` | oneMKL Sparse BLAS | FP64、1024x1024 Poisson格子、1始まりCSR |
| [cuSOLVER](cusolver/README.ja.md) | `solver_cpu` / `solver_gpu` / `openacc_cusolver` | Fortran DGESV、oneMKL | FP64、N=1024、右辺16本、正解1 |
| [cuRAND](curand/README.ja.md) | `rand_cpu` / `rand_gpu` / `openacc_curand` | `random_seed` / `random_number` | FP64、2^24個、seed 1234 |
| [Thrust](thrust/README.ja.md) | `reduce_cpu` / `reduce_gpu` / `openacc_thrust` | `sum(values*values)` | FP64、2^24個の1 |

教材は`<library>/examples/<stem>.f90`、測定用は
`<library>/benchmarks/<stem>_bench.f90`です。測定本体は同じディレクトリの
`*_workload.F90`にあり、方式別に独立してコンパイルします。
target名と実行ファイル名はstemと一致し、C/C++とはbuild treeで分離します。

[共通Fortran処理](../../common/fortran)は数学helperと、既存CLI・単調時計・
厳密な結果出力へ接続するC相互運用部分です。計算はFortranで行い、
C/C++のbenchmark実行ファイルを起動しません。
GPU Thrustだけは提供された[CUDA C++関数](thrust/examples/thrust_wrapper.cu)を
直接版とOpenACC版で共用します。

<a id="build-requirements"></a>

## 必要環境とCPU-onlyビルド

CMake 3.20以降、C17/C++17、Fortran 2008の機能を使用します。
CPU-onlyはGNU FortranまたはNVHPC、CUDA Fortran/OpenACCは
`nvfortran`が必要です。25.11公式interfaceと静的照合していますが、
NVHPCでのビルド済みという意味ではありません。ThrustにはNVCCと
同じToolkitのThrust/CCCLも必要です。GNU Fortranはdevice拡張を扱えません。
コンパイラ・profile・言語が異なるbuild treeや`.mod`を流用しないでください。

FFTWはFP32の`fftw3f`と`fftw3.f03`、threaded benchmarkには
`fftw3f_threads`が必要です。oneMKLはLP64 Fortran BLAS/LAPACK ABI、
`mkl_rt`、`mkl_service.h`、Sparse BLAS用`mkl_spblas.f90`が必要です。
moduleソースを選択コンパイラでビルドするため、他コンパイラの`.mod`はコピーしません。
実際の`MKLROOT`、必要なら`FFTW_ROOT`を設定してください。
依存不足・compile/link probe失敗は該当targetだけを無効化し、代替backendを黙って使いません。

既定の`GPU_SUITE_SOURCE_LANGUAGE=c-cpp`ではFortranコンパイラは不要です。
新しいtreeで明示的に選びます。

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/fortran-cpu \
  -DGPU_SUITE_SOURCE_LANGUAGE=fortran -DCMAKE_Fortran_COMPILER=gfortran \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_TESTS=ON
cmake --build /tmp/gpu-library-suite-local-build/fortran-cpu --parallel 2
cmake --build /tmp/gpu-library-suite-local-build/fortran-cpu \
  --target gpu_suite_partial_manifest --parallel 2
ctest --test-dir /tmp/gpu-library-suite-local-build/fortran-cpu --output-on-failure
```

FFTW/oneMKLがなくても組込み乱数・平方和のCPU版は使えます。
configure成功と全target有効化・実行成功は別です。
`fortran-probes/`のログ、`compile_commands.json`、verbose build出力を確認します。

<a id="separate-gpu-build-trees"></a>

## GPU用の分離ビルド

以下はリポジトリrootから、正規に使用できるGPU build/run環境で人間が実行する手順です。
Pegasusでは先に[Fortran初回実機確認](../../docs/PEGASUS_EXECUTION.md#fortran-first-validation)
に従います。ログインノードでのGPU処理は想定しません。
Toolkit root、CMake architecture、NVHPC targetは実環境の確定値を与えます。

```bash
: "${CUDA_TOOLKIT_ROOT:?Set the installed external CUDA Toolkit root}"
: "${CUDA_ARCHITECTURES:?Set the CMake CUDA architecture}"
: "${NVHPC_GPU_TARGET:?Set the NVHPC GPU target}"
export CUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_HOME="$CUDA_TOOLKIT_ROOT" CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset NVCOMPILER_FPU_STATE
mkdir -p build results
BUILD_ROOT="$(mktemp -d "$PWD/build/fortran.XXXXXX")"
RUN_DIR="$(mktemp -d "$PWD/results/fortran-pilot.XXXXXX")"
export CPU_CUDA_BUILD="$BUILD_ROOT/cpu-cuda" OPENACC_BUILD="$BUILD_ROOT/openacc"
set -o noclobber
SOURCE_HASH="$(python3 tools/hash_source_snapshot.py "$PWD")"
printf '%s\n' "$SOURCE_HASH" > "$RUN_DIR/source-snapshot.sha256"

cmake -S . -B "$CPU_CUDA_BUILD" -G "Unix Makefiles" \
  -DGPU_SUITE_SOURCE_LANGUAGE=fortran -DCMAKE_Fortran_COMPILER=nvfortran \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET"
cmake --build "$CPU_CUDA_BUILD" --target gpu_suite_partial_manifest --parallel 2 --verbose

cmake -S . -B "$OPENACC_BUILD" -G "Unix Makefiles" \
  -DGPU_SUITE_SOURCE_LANGUAGE=fortran -DCMAKE_Fortran_COMPILER=nvfortran \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF -DGPU_SUITE_BUILD_OPENACC=ON \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET"
cmake --build "$OPENACC_BUILD" --target gpu_suite_partial_manifest --parallel 2 --verbose
test "$SOURCE_HASH" = "$(python3 tools/hash_source_snapshot.py "$PWD")"
python3 tools/merge_manifests.py --output "$RUN_DIR/executables.json" \
  "$CPU_CUDA_BUILD/partial-manifest.json" "$OPENACC_BUILD/partial-manifest.json"
```

失敗したコマンドで止め、stdout/stderr、module/compiler version、metadata/manifestを
新規検証ディレクトリへ保存します。snapshot後にソースを変えたり再ビルドしたりしないでください。
snapshotにはignore対象でない未追跡ソースも含みます。
今回の未commit版では、prepare・runner・renderに同じ
`PROVENANCE_ARGS=(--source-snapshot-sha256 "$SOURCE_HASH")`を使用します。
将来のclean commitからのbuildなら空配列です。

両buildのCMake Toolkit rootと`nvhpc_cuda_home`の一致を確認します。
完全な構成はmanifest 36件（教材18＋benchmark 18）です。
一部依存が無くconfigureが通った状態を、6ライブラリの確認完了とはしません。

<a id="numerical-and-cpu-runtime-conditions"></a>

## 数値・CPU並列化の条件

新規FortranのRelease既定値は`-O3`です。NVHPC既定の`-fast`を採用しません。
targetにはGNUの`-fno-fast-math -ffp-contract=off`、またはNVHPCの
`-Kieee -Mnoflushz -Mnodaz`を適用します。GPUにはSeparate Memory Modeを明示し、
OpenACCには`-acc=gpu`を追加します。未承認のglobal fast-math、
自動並列化・自動GPU offloadをこのprofileで拒否します。
C/C++ bridgeにもunsafeなglobal flagsを許可しないため、手順はGCC/G++を使用します。
数値target optionsはglobal configure flagsとは別にbuild metadataへ記録します。
Releaseでも明示的verificationは残り、assertの有効無効には依存しません。

NVHPCのFPU設定を上書きできる`NVCOMPILER_FPU_STATE`は実行前にunsetします。
NVHPC benchmarkは非空の設定を拒否します。この条件から過去のC/C++測定時の
FPU状態を確定したり、[既存の数値条件の未確認事項](../../docs/PORTABILITY.ja.md#numerical-portability)
が解消したと説明したりしません。

FFTWの教材は逐次です。threaded benchmarkはplan作成前にthreads APIを設定します。
oneMKLは内部並列化し、要求数と観測された実効数を区別します。
組込み乱数・sumは自動並列化/offloadを有効にせず、
`cpu-fortran-random-serial`と`cpu-fortran-sum-serial`、実効1です。
乱数アルゴリズムはコンパイラ依存なのでcompiler identityも記録します。
C++ MT19937やGPU cuRANDと数列一致を要求しません。
初期pilotの要求1 threadは診断用で、本測定条件の承認ではありません。

<a id="first-checks-and-own-measurements"></a>

## 最初の確認と自己測定

各ライブラリREADMEにhelperを含む直接コンパイル、教材の実行パス、
小規模benchmarkコマンドがあります。教材は終了0と`verification PASS`を確認します。
教材の定数は保持し、benchmarkは実行時サイズを受け取ります。
まずcuBLASのCPU/CUDA/OpenACC×compute/E2Eを小規模に確認します。
単独実行の診断行は完全なcampaignデータではなく、runnerのrawに混ぜません。

測定基盤は[既存の自己測定手順](../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)
を再利用し、上のFortran build/manifest、新規run IDと次の入力に置き換えます。

```bash
MANIFEST="$RUN_DIR/executables.json"
cp configs/fortran/pilot.json "$RUN_DIR/config-input.json"
CONFIG_INPUT="$RUN_DIR/config-input.json"
CONFIG="$RUN_DIR/effective-config.json"
PROVENANCE_ARGS=(--source-snapshot-sha256 "$SOURCE_HASH")
```

`configs/fortran/pilot.json`は各ライブラリ3つの小規模診断ケース、2 trials、
両scopeを持ちます。問題・精度・verificationを維持し、
`source_language="fortran"`とFortran組込みbackend名を使用します。
過去のC/C++ publication workloadとは別で、Fortran本測定profileは未確定です。
最初の6行smokeでは、新規コピーを編集し、cuBLASだけ有効、最初のcaseだけ、
`run_mode="smoke"`、両scopeのwarmup/repeat/trialsを1にします。
他5ライブラリのobjectは削除せずdisabledにし、正本configを上書きしません。

その後、既存手順の[effective config正規化](../../docs/PORTABILITY.ja.md#2-make-a-small-measurement-check-then-choose-the-suite-settings)、
[runtime取得・prepare・runner](../../docs/PORTABILITY.ja.md#3-capture-the-real-runtime-environment-and-launch-one-local-block)、
[検証・集計](../../docs/PORTABILITY.ja.md#validate-and-aggregate-a-completed-run)、
[自分のデータから2パネル6図生成](../../docs/PORTABILITY.ja.md#figures-from-your-own-measurements)
へ進みます。module環境では実際のmodule listを取得します。
上のdirty-source引数を、clean用サンプルの空配列で上書きしないでください。
`--local`手順は本当の単一ホスト非PBS用です。PegasusではPBS情報を消すのではなく、
既存PBS launcher/provenance手順を使います。

raw schema version 1を維持し、Fortran行は
`parameters.source_language="fortran"`、manifestは`compiler_language="fortran"`です。
古いC/C++ rawは変換しません。混合言語manifest/run・同一図を拒否し、
集計parameter signatureに言語を残します。凡例にもFortranと組込みbackendを表示します。
6図には3ケース、両scope、3方式と、数値検証成功、
CUDA/OpenACC Thrustの同一library versionが必要です。6行smokeは6図の入力にはなりません。
合成テスト値や既存C/C++値をFortran実測へ流用せず、性能順位を先に決めません。

<a id="teaching-extraction-changes-and-validation-limits"></a>

## 掲載抽出版からの変更と検証範囲

| ページ／ライブラリ | 配布用に必要な補完・修正 |
| --- | --- |
| 9、12／cuFFT | allocation・plan・APIチェックとfinite/DC/non-DC検証。主例のplan・同期方式は維持 |
| 18、20／cuBLAS | allocation/APIチェックと全要素1のDGEMM検証。初期CとFP64を維持 |
| 25–30／cuSPARSE | 省略されていた1始まりPoisson helperを追加。descriptor全フィールド初期化と結果検証 |
| 35–38／cuSOLVER | A=N*I+ones、B=2Nのhelperを追加。教材ではgetrf結果を別infoで確認後にsolveし、解・残差を検証 |
| 43、45／cuRAND | allocation/APIと分布検証。組込みRNGと指定cuRAND generatorは維持 |
| 50–53／Thrust | finite・平方和検証。共通C++関数で例外を捕捉し、NaNを返して呼出側で失敗にする |

p.13の省略記号を含むstream補足はbuild対象外です。
同期方式を全主例に置き換えていません。参照抽出ファイルはignore対象の
ローカル検証入力に原文のまま保持し、PPTXは編集・独立確認していません。

GNU Fortranで組込みCPU版とhelperの数学を確認しています。
テスト用代替providerで外部CPU4経路、両scope、状態復元、NaN/Inf・info失敗も
確認していますが、実oneMKL/FFTWやNVHPC ABIの検証ではありません。
実行コマンドと未実行項目は[検証報告](../../docs/VALIDATION_REPORT.md#fortran-local-validation)
を参照してください。NVHPC実コンパイル、実FFTW/oneMKLリンク、GPU全例・benchmark、
compute-sanitizer、Pegasusのruntime/provenance確認は実性能測定の前に必要です。

公式照合先：
[NVIDIA Fortran CUDA Interfaces 25.11](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/fortran-cuda-interfaces/index.html)、
[CUDA Fortran Guide 25.11](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/cuda-fortran-prog-guide/index.html)、
[HPC Compilers Reference 25.11](https://docs.nvidia.com/hpc-sdk/archive/25.11/compilers/hpc-compilers-ref-guide/index.html)。
静的な照合をGPU実行確認と扱わないでください。
