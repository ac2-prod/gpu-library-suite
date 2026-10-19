# Thrust C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

transform-reduceで`sum(values[i]^2)`を計算します。変換と集約を組み合わせ、二乗値の中間配列を保存せずに求めます。

<a id="teaching-default"></a>

## 教材の既定問題

入力は`2^24`個のFP64の1で、二乗和の期待値は`16777216`です。CPUは逐次STLの`std::transform_reduce`です。hostへ返す結果はdouble 1個です。

<a id="dependencies"></a>

## 依存環境

逐次CPUはC++17と標準ライブラリだけで動きます。GPUにはThrust／CCCLとCUDAが必要です。

GPUはCUDA Toolkitと対応GPU／ドライバ、OpenACCはさらにNVHPC nvc++が必要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=thrust
CPU_TARGET=reduce_cpu
GPU_TARGET=reduce_gpu
DIRECTIVE_TARGET=openacc_thrust
CPU_OPTIONS=(-DGPU_SUITE_CPU_THRUST_BACKEND=STL)
```

<a id="canonical-sources-and-targets"></a>

## 正本ソースとターゲット

| 用途 | ソース | CMakeターゲット／実行ファイル |
| --- | --- | --- |
| CPU教材 | `examples/reduce_cpu.cpp` | `reduce_cpu` |
| CUDA教材 | `examples/reduce_gpu.cu` | `reduce_gpu` |
| OpenACC教材 | `examples/openacc_thrust.cpp` | `openacc_thrust` |
| CPUベンチマーク | `benchmarks/reduce_cpu_bench.cpp` | `reduce_cpu_bench` |
| CUDAベンチマーク | `benchmarks/reduce_gpu_bench.cu` | `reduce_gpu_bench` |
| OpenACCベンチマーク | `benchmarks/openacc_thrust_bench.cpp` | `openacc_thrust_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

変換関数・集約演算・単位元を一組で変更します。浮動小数点の集約は順序で結果が変わり得るため、一般入力では適切な誤差許容値を使います。繰り返すGPU処理では入力を常駐させます。ディレクティブが管理するpointerをwrapperで解放せず、確保した側に所有権を残します。

CUDA版はdevice配列を直接管理し、OpenACC版はdata領域と`host_data use_device`を使って同じGPUライブラリへ渡します。配列の所有権を混ぜず、結果を読む前に完了を保証してください。

<a id="direct-compile"></a>

## 直接コンパイル

CMakeを使わない場合の代替手順です。リポジトリ直下のBashで、必要なheader・libraryの検索パスが設定済みであることを前提とします。各方式のblockを選び、成功した場合だけ実行します。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp -o "$DIRECT_BUILD/reduce_cpu" &&
"$DIRECT_BUILD/reduce_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" -I"${CUDA_CCCL_INCLUDE_DIR:?Set the directory containing thrust headers}" \
  nvidia/c-cpp/thrust/examples/reduce_gpu.cu \
  -o "$DIRECT_BUILD/reduce_gpu" &&
"$DIRECT_BUILD/reduce_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" -I"${CUDA_CCCL_INCLUDE_DIR:?Set the directory containing thrust headers}" \
  nvidia/c-cpp/thrust/examples/openacc_thrust.cpp \
  -o "$DIRECT_BUILD/openacc_thrust" &&
"$DIRECT_BUILD/openacc_thrust"
```

<a id="cmake-configure-and-build"></a>

## CMakeの構成とビルド

上の選択値を[共通手順](../../../docs/BUILDING.ja.md)へ渡します。CPU、CUDA、OpenACCの新しいtreeを用い、選択したtargetだけを構築します。CPU数値ライブラリが不要ならGPUの節だけを利用できます。

<a id="first-build-and-run"></a>

## 最初のビルドと実行

選択・build・実行の順に進み、終了値0と「教材の既定問題」の期待値を確認してください。欠落targetや非ゼロ終了を成功とは扱いません。共有計算機では計算ノードの割当内で実行します。

<a id="run-the-examples"></a>
Fortran教材は[別の案内](../../fortran/thrust/README.ja.md)を使います。

<a id="from-example-to-benchmark"></a>

## サンプルからベンチマークへの対応

教材に時間測定用の引数はありません。性能測定では対応する`_bench`を使い、library（内部scope名`compute`）とend-to-endを区別します。以下は既存benchmarkの参照情報です。[測定手順](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)でbenchmarkを有効にしたbuildを作り、そこで設定したパスを使ってください。最小教材buildにはbenchmarkは含まれません。

<a id="benchmark-cli"></a>

## ベンチマークCLI

以下はリンク先の測定手順で作ったbuildを使う1スレッド要求の小規模確認です。BLAS／Sparse／Solverはその手順のoneMKL構成に対応します。別providerでbuildした場合は`--cpu-backend`もbuildに合わせて変更してください。

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/thrust/reduce_cpu_bench" \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --abs-tolerance 1e-12 --rel-tolerance 1e-10 \
  --output - --format jsonl --cpu-backend cpu-stl-serial \
  --cpu-threads 1 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

production実行は閾値とroleを実効設定から取得し、`tools/run_suite.py`だけがノードraw-resultファイルを書き込みます。

<a id="timing-scopes"></a>

## 測定スコープ

表示名libraryの内部scopeは`compute`です。

`compute`は、測定前に入力／デバイスメモリを準備し、transform-reduceの反復だけを測定します。GPUの`end-to-end`は各repeatのデバイスメモリ作成、入力転送、reduction、完了／結果取得を含み、後処理は終了時刻の後に行います。CPUベクトルと正本の全要素1のホスト入力は測定区間外で準備し、CPU経路にはデバイス転送はありません。正本状態の復元と検証も測定区間外です。

<a id="verification"></a>

## 検証

測定結果を正本の解析的な二乗和と、`abs_tolerance + rel_tolerance * reference_scale`を使って比較します。検証失敗は保持しますが、状態復元後の試行を妨げる致命的失敗とは扱いません。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCが入力メモリの確保と転送を所有し、その後`thrust::device_pointer_cast`で既知のデバイスアドレスを変換します。Thrustの同期アルゴリズムは、通常ホストで結果を取得するための追加の完了待ちを必要としません。教材のOpenACCサンプルでは、連携の境界を示すために明示的な`cudaDeviceSynchronize`を残しています。コンパイルとリンクの両方で、NVHPC CUDA C++連携フラグを指定します。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

保存済みCPU評価は逐次です。要求スレッド数が多くても並列版にはなりません。自分のマシンのCPU名・選択したbackend・要求値と実効値を区別して記録します。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

PegasusでNVIDIA C/C++のbuild・数値smoke・性能測定の保存記録があります。[版と確認範囲](../../../docs/VALIDATION_REPORT.md#current-coexistence-status)を参照してください。ローカルfixture検査と実機検証は別で、別環境や後続変更の動作を保証しません。数値誤差・乱数分布の検証条件とCPU並列性の制約は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。
