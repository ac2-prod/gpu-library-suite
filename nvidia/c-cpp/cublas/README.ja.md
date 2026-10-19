# cuBLAS C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

FP64 DGEMMで密行列積`C = alpha*A*B + beta*C`を計算します。CPU BLASからGPU直接呼び出し・ディレクティブによる配列管理への対応を確認できます。

<a id="teaching-default"></a>

## 教材の既定問題

列優先の正方行列で`m=n=k=1024`、`alpha=beta=1`、A/B/Cはすべて1です。出力は全要素`1025`となり、C/C++教材は最大誤差`1e-10`以下で成功します。

<a id="dependencies"></a>

## 依存環境

CPUではoneMKL・OpenBLAS・generic CBLASのいずれかを明示選択します。以下の例はOpenBLAS（`cblas.h`・`libopenblas`）を選びます。

GPUはCUDA Toolkitと対応GPU／ドライバ、OpenACCはさらにNVHPC nvc++が必要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cublas
CPU_TARGET=blas_cpu
GPU_TARGET=blas_gpu
DIRECTIVE_TARGET=openacc_cublas
CPU_OPTIONS=(-DGPU_SUITE_CPU_BLAS_BACKEND=OPENBLAS)
```

<a id="canonical-sources-and-targets"></a>

## 正本ソースとターゲット

| 用途 | ソース | CMakeターゲット／実行ファイル |
| --- | --- | --- |
| CPU教材 | `examples/blas_cpu.c` | `blas_cpu` |
| CUDA教材 | `examples/blas_gpu.cu` | `blas_gpu` |
| OpenACC教材 | `examples/openacc_cublas.cpp` | `openacc_cublas` |
| CPUベンチマーク | `benchmarks/blas_cpu_bench.c` | `blas_cpu_bench` |
| CUDAベンチマーク | `benchmarks/blas_gpu_bench.cu` | `blas_gpu_bench` |
| OpenACCベンチマーク | `benchmarks/openacc_cublas_bench.cpp` | `openacc_cublas_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

配列の格納順とleading dimension（`lda`、`ldb`、`ldc`）を合わせます。betaが0でないためCも入力です。A/Bだけをコピーしてはいけません。繰り返す場合は配列とhandleを適切な寿命で保持します。CPUの並列性は選択したBLAS providerが担い、要求スレッド数は実測worker数ではありません。

CUDA版はdevice配列を直接管理し、OpenACC版はdata領域と`host_data use_device`を使って同じGPUライブラリへ渡します。配列の所有権を混ぜず、結果を読む前に完了を保証してください。

<a id="direct-compile"></a>

## 直接コンパイル

CMakeを使わない場合の代替手順です。リポジトリ直下のBashで、必要なheader・libraryの検索パスが設定済みであることを前提とします。各方式のblockを選び、成功した場合だけ実行します。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 nvidia/c-cpp/cublas/examples/blas_cpu.c -lopenblas -lm -o "$DIRECT_BUILD/blas_cpu" &&
"$DIRECT_BUILD/blas_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cublas/examples/blas_gpu.cu -lcublas \
  -o "$DIRECT_BUILD/blas_gpu" &&
"$DIRECT_BUILD/blas_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cublas/examples/openacc_cublas.cpp -cudalib=cublas \
  -o "$DIRECT_BUILD/openacc_cublas" &&
"$DIRECT_BUILD/openacc_cublas"
```

<a id="cmake-configure-and-build"></a>

## CMakeの構成とビルド

上の選択値を[共通手順](../../../docs/BUILDING.ja.md)へ渡します。CPU、CUDA、OpenACCの新しいtreeを用い、選択したtargetだけを構築します。CPU数値ライブラリが不要ならGPUの節だけを利用できます。

<a id="first-build-and-run"></a>

## 最初のビルドと実行

選択・build・実行の順に進み、終了値0と「教材の既定問題」の期待値を確認してください。欠落targetや非ゼロ終了を成功とは扱いません。共有計算機では計算ノードの割当内で実行します。

<a id="run-the-examples"></a>
Fortran教材は[別の案内](../../fortran/cublas/README.ja.md)を使います。

<a id="from-example-to-benchmark"></a>

## サンプルからベンチマークへの対応

教材に時間測定用の引数はありません。性能測定では対応する`_bench`を使い、library（内部scope名`compute`）とend-to-endを区別します。以下は既存benchmarkの参照情報です。[測定手順](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)でbenchmarkを有効にしたbuildを作り、そこで設定したパスを使ってください。最小教材buildにはbenchmarkは含まれません。

<a id="benchmark-cli"></a>

## ベンチマークCLI

以下はリンク先の測定手順で作ったbuildを使う1スレッド要求の小規模確認です。BLAS／Sparse／Solverはその手順のoneMKL構成に対応します。別providerでbuildした場合は`--cpu-backend`もbuildに合わせて変更してください。

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_cpu_bench" \
  --size 128 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

選択したプロバイダに対応するCPUバックエンド名を使ってください。有効スレッド数が独立に確認できている場合は、`--cpu-threads-effective`で渡します。それ以外はnullのままにします。productionスイート実行の閾値・roleは、すべて実効設定から取得する必要があります。

<a id="timing-scopes"></a>

## 測定スコープ

表示名libraryの内部scopeは`compute`です。

`compute`は、メモリ／ハンドルの作成とCの復元を測定開始前に行い、repeatを通してDGEMMの更新を連続適用します。そのため検証では、repeat回の更新を反映した結果を確認します。`end-to-end`は各repeatのメモリ確保、コピー、ハンドルのライフサイクル、DGEMM、完了、結果取得を測定し、後処理は終了時刻の後に行います。各repeatの前に正本のCを復元するため、状態を引き継ぎません。

<a id="verification"></a>

## 検証

実効設定を使い、最大絶対誤差を`abs_tolerance + rel_tolerance * reference_scale`と比較します。検証失敗はraw失敗として保持し、状態を復元した後の試行は継続できます。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCデータ領域がA、B、Cを所有します。`host_data use_device`でそのアドレスをcuBLASへ渡し、別のCUDAメモリ確保が配列を重複所有することはありません。正本のリンク対象は`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::cublas`です。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

測定では選択したproduction CPU backendを明示し、依存不足による別providerへの置換を行いません。スレッド要求と実効worker数は区別します。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

PegasusでNVIDIA C/C++のbuild・数値smoke・性能測定の保存記録があります。[版と確認範囲](../../../docs/VALIDATION_REPORT.md#current-coexistence-status)を参照してください。ローカルfixture検査と実機検証は別で、別環境や後続変更の動作を保証しません。数値誤差・乱数分布の検証条件とCPU並列性の制約は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。
