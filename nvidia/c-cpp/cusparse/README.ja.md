# cuSPARSE C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

CSR形式のFP64疎行列ベクトル積`y = alpha*A*x + beta*y`を計算します。教材行列は規則的な2次元5点Poisson stencilです。

<a id="teaching-default"></a>

## 教材の既定問題

1024×1024格子から1,048,576行の行列を生成します。対角4、隣接−1、`alpha=beta=1`、`x=y=1`です。出力yは内部1・辺2・角3となります。CSR添字は0始まりです。

<a id="dependencies"></a>

## 依存環境

CPU教材はoneMKL Sparse LP64（`mkl_spblas.h`・`mkl_rt`）を使います。REFERENCEは補助benchmark用で、教材providerを代替しません。

GPUはCUDA Toolkitと対応GPU／ドライバ、OpenACCはさらにNVHPC nvc++が必要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cusparse
CPU_TARGET=sparse_cpu
GPU_TARGET=sparse_gpu
DIRECTIVE_TARGET=openacc_cusparse
CPU_OPTIONS=(-DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL)
```

<a id="canonical-sources-and-targets"></a>

## 正本ソースとターゲット

| 用途 | ソース | CMakeターゲット／実行ファイル |
| --- | --- | --- |
| CPU教材 | `examples/sparse_cpu.c` | `sparse_cpu` |
| CUDA教材 | `examples/sparse_gpu.cu` | `sparse_gpu` |
| OpenACC教材 | `examples/openacc_cusparse.cpp` | `openacc_cusparse` |
| CPUベンチマーク | `benchmarks/sparse_cpu_bench.c` | `sparse_cpu_bench` |
| CUDAベンチマーク | `benchmarks/sparse_gpu_bench.cu` | `sparse_gpu_bench` |
| OpenACCベンチマーク | `benchmarks/openacc_cusparse_bench.cpp` | `openacc_cusparse_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

CSRのrow pointer・column index・値を一組として置き換え、添字幅・基準・次元・非零要素の順序を一致させます。betaが0でなければ初期yも必要です。descriptorとanalysisは保持する行列・pointerについてAPIが許す範囲で再利用します。規則的な5点行列の性能を不規則な実行列へ一般化しないでください。

CUDA版はdevice配列を直接管理し、OpenACC版はdata領域と`host_data use_device`を使って同じGPUライブラリへ渡します。配列の所有権を混ぜず、結果を読む前に完了を保証してください。

<a id="direct-compile"></a>

## 直接コンパイル

CMakeを使わない場合の代替手順です。リポジトリ直下のBashで、必要なheader・libraryの検索パスが設定済みであることを前提とします。各方式のblockを選び、成功した場合だけ実行します。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 -I"${MKLROOT:?Set the installed oneMKL prefix}/include" \
  nvidia/c-cpp/cusparse/examples/sparse_cpu.c -L"${MKL_LIBDIR:?Set the oneMKL library directory}" \
  -lmkl_rt -lpthread -ldl -lm -o "$DIRECT_BUILD/sparse_cpu" &&
"$DIRECT_BUILD/sparse_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cusparse/examples/sparse_gpu.cu -lcusparse \
  -o "$DIRECT_BUILD/sparse_gpu" &&
"$DIRECT_BUILD/sparse_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp -cudalib=cusparse \
  -o "$DIRECT_BUILD/openacc_cusparse" &&
"$DIRECT_BUILD/openacc_cusparse"
```

<a id="cmake-configure-and-build"></a>

## CMakeの構成とビルド

上の選択値を[共通手順](../../../docs/BUILDING.ja.md)へ渡します。CPU、CUDA、OpenACCの新しいtreeを用い、選択したtargetだけを構築します。CPU数値ライブラリが不要ならGPUの節だけを利用できます。

<a id="first-build-and-run"></a>

## 最初のビルドと実行

選択・build・実行の順に進み、終了値0と「教材の既定問題」の期待値を確認してください。欠落targetや非ゼロ終了を成功とは扱いません。共有計算機では計算ノードの割当内で実行します。

<a id="run-the-examples"></a>
Fortran教材は[別の案内](../../fortran/cusparse/README.ja.md)を使います。

<a id="from-example-to-benchmark"></a>

## サンプルからベンチマークへの対応

教材に時間測定用の引数はありません。性能測定では対応する`_bench`を使い、library（内部scope名`compute`）とend-to-endを区別します。以下は既存benchmarkの参照情報です。[測定手順](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)でbenchmarkを有効にしたbuildを作り、そこで設定したパスを使ってください。最小教材buildにはbenchmarkは含まれません。

<a id="benchmark-cli"></a>

## ベンチマークCLI

以下はリンク先の測定手順で作ったbuildを使う1スレッド要求の小規模確認です。BLAS／Sparse／Solverはその手順のoneMKL構成に対応します。別providerでbuildした場合は`--cpu-backend`もbuildに合わせて変更してください。

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusparse/sparse_cpu_bench" \
  --size 4096 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

平方数の`--size`から、`nx`、`ny`、厳密な非零要素数を決定します。スイート実行では設定が定めるroleと閾値を渡し、raw-resultファイルの書込み主体をランナーだけに限定します。

<a id="timing-scopes"></a>

## 測定スコープ

表示名libraryの内部scopeは`compute`です。

`compute`は、測定前に記述子とメモリを準備し、yを復元してから、repeatを通してSpMVの状態更新を連続適用します。検証は全repeatを反映した出力を確認します。`end-to-end`は、各repeatのメモリ、転送、記述子作成、作業領域の照会／確保、SpMV、完了、copyoutを測定します。後処理は終了時刻の後に行い、各repeatの前に正本のyを復元します。

<a id="verification"></a>

## 検証

最大絶対誤差を`abs_tolerance + rel_tolerance * reference_scale`と比較します。検証失敗は復旧可能な試行の失敗です。残りの試行をskipするのは、致命的な記述子、メモリ確保、API、clockの失敗だけです。

<a id="openacc-notes"></a>

## OpenACCの注意点

データ領域がCSRとベクトルのメモリを所有し、`host_data use_device`がそのアドレスをcuSPARSEへ渡します。CUDAで確保するのはサイズが0でないのcuSPARSE作業領域だけで、0バイトの作業領域は`nullptr`のままです。ターゲットは`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::cusparse`をリンクします。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

測定では選択したproduction CPU backendを明示し、依存不足による別providerへの置換を行いません。スレッド要求と実効worker数は区別します。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

PegasusでNVIDIA C/C++のbuild・数値smoke・性能測定の保存記録があります。[版と確認範囲](../../../docs/VALIDATION_REPORT.md#current-coexistence-status)を参照してください。ローカルfixture検査と実機検証は別で、別環境や後続変更の動作を保証しません。数値誤差・乱数分布の検証条件とCPU並列性の制約は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。
