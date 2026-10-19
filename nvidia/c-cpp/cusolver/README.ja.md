# cuSOLVER C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

LU分解を使って複数右辺のFP64密行列方程式`A*X=B`を解きます。行列のほかpivot配列とsolverの終了情報を扱う流れを示します。

<a id="teaching-default"></a>

## 教材の既定問題

`N=1024`、`nrhs=16`、`A=N*I+ones`、Bの全要素は`2*N`です。解はすべて1となります。APIの成功と`info=0`に加え、数値検証の成功が必要です。

<a id="dependencies"></a>

## 依存環境

CPUはLAPACKE APIが必要で、oneMKL・LAPACKEを含むOpenBLAS・generic LAPACKEを明示選択します。以下の例はgeneric LAPACKEです。BLASだけの環境では不足します。

GPUはCUDA Toolkitと対応GPU／ドライバ、OpenACCはさらにNVHPC nvc++が必要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cusolver
CPU_TARGET=solver_cpu
GPU_TARGET=solver_gpu
DIRECTIVE_TARGET=openacc_cusolver
CPU_OPTIONS=(-DGPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE)
```

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

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

分解でA、solveでBが上書きされるため、残差確認や再試行に必要な原本を保持します。API失敗とinfoによる特異性の報告を区別してください。Aが同じでBだけ変わるなら、LUとpivotを保持してgetrsへ渡せます。毎回分解する場合とは費用が異なります。

CUDA版はdevice配列を直接管理し、OpenACC版はdata領域と`host_data use_device`を使って同じGPUライブラリへ渡します。配列の所有権を混ぜず、結果を読む前に完了を保証してください。

<a id="direct-compile"></a>

## 直接コンパイル

CMakeを使わない場合の代替手順です。リポジトリ直下のBashで、必要なheader・libraryの検索パスが設定済みであることを前提とします。各方式のblockを選び、成功した場合だけ実行します。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 nvidia/c-cpp/cusolver/examples/solver_cpu.c \
  -llapacke -llapack -lblas -lm -o "$DIRECT_BUILD/solver_cpu" &&
"$DIRECT_BUILD/solver_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cusolver/examples/solver_gpu.cu -lcusolver \
  -o "$DIRECT_BUILD/solver_gpu" &&
"$DIRECT_BUILD/solver_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp -cudalib=cusolver \
  -o "$DIRECT_BUILD/openacc_cusolver" &&
"$DIRECT_BUILD/openacc_cusolver"
```

<a id="cmake-configure-and-build"></a>

## CMakeの構成とビルド

上の選択値を[共通手順](../../../docs/BUILDING.ja.md)へ渡します。CPU、CUDA、OpenACCの新しいtreeを用い、選択したtargetだけを構築します。CPU数値ライブラリが不要ならGPUの節だけを利用できます。

<a id="first-build-and-run"></a>

## 最初のビルドと実行

選択・build・実行の順に進み、終了値0と「教材の既定問題」の期待値を確認してください。欠落targetや非ゼロ終了を成功とは扱いません。共有計算機では計算ノードの割当内で実行します。

<a id="run-the-examples"></a>
Fortran教材は[別の案内](../../fortran/cusolver/README.ja.md)を使います。

<a id="from-example-to-benchmark"></a>

## サンプルからベンチマークへの対応

教材に時間測定用の引数はありません。性能測定では対応する`_bench`を使い、library（内部scope名`compute`）とend-to-endを区別します。以下は既存benchmarkの参照情報です。[測定手順](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)でbenchmarkを有効にしたbuildを作り、そこで設定したパスを使ってください。最小教材buildにはbenchmarkは含まれません。

<a id="benchmark-cli"></a>

## ベンチマークCLI

以下はリンク先の測定手順で作ったbuildを使う1スレッド要求の小規模確認です。BLAS／Sparse／Solverはその手順のoneMKL構成に対応します。別providerでbuildした場合は`--cpu-backend`もbuildに合わせて変更してください。

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusolver/solver_cpu_bench" \
  --size 64 --nrhs 2 --warmup 1 --repeat 1 --trials 1 \
  --scope end-to-end --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

cuSOLVERのすべてのスコープで`repeat=1`が必要です。選択したLAPACKEプロバイダに対応するバックエンド名を使い、ランタイムで有効スレッド数を確認できなければnullのままにしてください。productionスイートの閾値は実効設定だけから取得します。

<a id="timing-scopes"></a>

## 測定スコープ

表示名libraryの内部scopeは`compute`です。

`compute`は、測定前にメモリ確保、転送、作業領域照会を行い、正本の行列／右辺（RHS）を復元してから、`getrf`と`getrs`からなるパイプラインを1回測定します。`end-to-end`には、メモリ確保、コピーまたはOpenACC update、ハンドル/作業領域作成、求解、完了、解と両方のinfo codeの取得を含みます。後処理は終了時刻の後です。warm-up、状態復元、検証は測定区間外です。

<a id="verification"></a>

## 検証

effectiveな絶対・相対許容誤差を使い、解の相対誤差と相対残差の両方を確認します。`getrf_info`と`getrs_info`はそれぞれ0でなければなりません。OpenACCでは両方を`-1`で初期化し、ホストへ明示的にコピーするため、copybackの欠落を成功と誤認しません。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCデータ領域が行列、RHS、pivot、info配列を所有します。小さな`host_data use_device` 領域からそれらをcuSOLVERへ渡し、作業領域だけを手動でCUDAメモリ確保します。end-to-endの結果取得では、測定終了前にRHSと両info配列へ明示的な`acc update self`を行います。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

測定では選択したproduction CPU backendを明示し、依存不足による別providerへの置換を行いません。スレッド要求と実効worker数は区別します。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

PegasusでNVIDIA C/C++のbuild・数値smoke・性能測定の保存記録があります。[版と確認範囲](../../../docs/VALIDATION_REPORT.md#current-coexistence-status)を参照してください。ローカルfixture検査と実機検証は別で、別環境や後続変更の動作を保証しません。数値誤差・乱数分布の検証条件とCPU並列性の制約は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。
