# cuSOLVER C/C++のサンプルとベンチマーク

[English](README.md) | [日本語](README.ja.md)

このディレクトリでは、密なFP64連立方程式の正本問題を、LAPACKE、直接cuSOLVER、OpenACC/cuSOLVERで解きます。方程式系の構成は[`docs/PROJECT_SPECIFICATION.md`（英語）](../../../docs/PROJECT_SPECIFICATION.md)、CLI、段階分割、時間測定、検証は[`docs/BENCHMARK_PROTOCOL.ja.md`](../../../docs/BENCHMARK_PROTOCOL.ja.md)が定めます。

<a id="canonical-sources-and-targets"></a>

## 正本ソースとターゲット

| 用途 | ソース | CMakeターゲット／実行ファイル |
| --- | --- | --- |
| CPU教材 | `examples/solver_cpu.c` | `solver_cpu` |
| CUDA教材 | `examples/solver_gpu.cu` | `solver_gpu` |
| OpenACC教材 | `examples/openacc_cusolver.cpp` | `openacc_cusolver` |
| CPUベンチマーク | `benchmarks/solver_cpu_bench.c` | `solver_cpu_bench` |
| CUDAベンチマーク | `benchmarks/solver_gpu_bench.cu` | `solver_gpu_bench` |
| OpenACCベンチマーク | `benchmarks/openacc_cusolver_bench.cpp` | `openacc_cusolver_bench` |

<a id="teaching-default"></a>

## 教材の既定問題

すべての教材版は、`n=1024`、`nrhs=16`のFP64正本問題を解きます。CPU教材サンプルでは、意図的に`LAPACKE_dgesv`を示します。ベンチマーク版は因子分解と求解を別の段階に分け、`getrf_info`と`getrs_info`を独立に確認できるようにしています。

<a id="dependencies"></a>

## 依存環境

CPUターゲットには、`LAPACKE_dgesv`、`LAPACKE_dgetrf`、`LAPACKE_dgetrs`のすべてをコンパイル・リンクできるONEMKL、OPENBLAS、GENERIC_LAPACKEのいずれかのプロバイダが必要です。プロバイダごとのキャッシュキーで、他プロバイダの古いパスの混入を防ぎます。CUDAプログラムにはCUDA RuntimeとcuSOLVER 密行列APIが必要です。OpenACCプログラムには、さらにNVHPC OpenACC C++が必要です。

<a id="direct-compile"></a>

## 直接コンパイル

CPUソースは、CMakeがプロバイダのヘッダを指定しない限り、既定で`<lapacke.h>`をincludeします。代表的な直接コンパイルのコマンドは次のとおりです。

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cusolver/examples/solver_cpu.c \
  -llapacke -llapack -lblas -lm \
  -o /tmp/gpu-library-suite-local-build/solver_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cusolver/examples/solver_gpu.cu \
  -lcusolver -o /tmp/gpu-library-suite-local-build/solver_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp -cudalib=cusolver \
  -o /tmp/gpu-library-suite-local-build/openacc_cusolver-direct
```

<a id="cmake-configure-and-build"></a>

## CMakeのconfigureとビルド

`CUDA_TOOLKIT_ROOT`、`CUDA_ARCHITECTURES`、`NVHPC_GPU_TARGET`に、サイトで承認された具体値が設定済みであることを前提とします。

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target solver_cpu solver_gpu solver_cpu_bench solver_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cusolver openacc_cusolver_bench
```

<a id="run-the-examples"></a>

## サンプルの実行

上記の対応ターゲットをビルドした後、リポジトリのルートから実行します。

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusolver/solver_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusolver/solver_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cusolver/openacc_cusolver"
printf 'OpenACC exit status: %s\n' "$?"
```

solverの`info`値、解の誤差、各終了ステータスを確認してください。ターゲットがない場合や、終了値が0以外の場合、比較は未完了です。CPU教材の呼び出しは`LAPACKE_dgesv`ですが、ベンチマークでは以下のように因子分解と求解を分離します。続いて[測定と結果処理](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)へ進んでください。

<a id="benchmark-cli"></a>

## ベンチマークCLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cusolver/solver_cpu_bench \
  --size 64 --nrhs 2 --warmup 1 --repeat 1 --trials 1 \
  --scope end-to-end --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-generic-lapacke --cpu-threads 48 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

cuSOLVERのすべてのスコープで`repeat=1`が必要です。選択したLAPACKEプロバイダに対応するバックエンド名を使い、ランタイムで有効スレッド数を確認できなければnullのままにしてください。productionスイートの閾値は実効設定だけから取得します。

<a id="timing-scopes"></a>

## 測定スコープ

`compute`は、測定前にメモリ確保、転送、作業領域照会を行い、正本の行列／右辺（RHS）を復元してから、`getrf`と`getrs`からなるパイプラインを1回測定します。`end-to-end`には、メモリ確保、コピーまたはOpenACC update、ハンドル/作業領域作成、求解、完了、解と両方のinfo codeの取得を含みます。後処理は終了時刻の後です。warm-up、状態復元、検証は測定区間外です。

<a id="verification"></a>

## 検証

effectiveな絶対・相対許容誤差を使い、解の相対誤差と相対残差の両方を確認します。`getrf_info`と`getrs_info`はそれぞれ0でなければなりません。OpenACCでは両方を`-1`で初期化し、ホストへ明示的にコピーするため、copybackの欠落を成功と誤認しません。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

設定したプロバイダがproductionのprimaryバックエンドで、名前は`cpu-onemkl`、`cpu-openblas`、`cpu-generic-lapacke`のいずれかです。プロバイダを暗黙に置き換えず、要求スレッド数を有効スレッド数とは仮定しません。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCデータ領域が行列、RHS、pivot、info配列を所有します。小さな`host_data use_device` 領域からそれらをcuSOLVERへ渡し、作業領域だけを手動でCUDAメモリ確保します。end-to-endの結果取得では、測定終了前にRHSと両info配列へ明示的な`acc update self`を行います。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

ハードウェア検出後にアルゴリズムの数値モードを切り替えることはなく、メモリ確保や因子分解の失敗をそのまま報告します。ローカルテストはfake LAPACKE/CUDA interfaceを使い、production用LAPACKEインストール環境、cuSOLVERランタイム、NVHPCコンパイラ、実GPU、Pegasus jobは実行していません。これらはローカルでは未検証です。その後に保存された実機実行記録は、[検証報告（英語）](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)でこれらのローカルテストと区別しています。
