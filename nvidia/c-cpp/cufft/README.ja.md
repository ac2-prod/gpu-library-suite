# cuFFT C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

長さ1,024のFP32複素数FFTを4,096本まとめてforward変換します。多数の信号をまとめて処理するアプリケーションで、batched FFTの呼び出し方を学べます。

<a id="teaching-default"></a>

## 教材の既定問題

out-of-placeの1D C2Cです。複素数をinterleaved形式で連続配置し、入力をすべて`1+0i`にします。各FFTのDC成分は`1024+0i`、他は0で、ソース内の許容誤差で確認します。

<a id="dependencies"></a>

## 依存環境

CPUには単精度FFTWの`fftw3.h`・`libfftw3f`が必要です。通常の探索範囲外なら`FFTW_ROOT`でprefixを指定します。スレッド版はbenchmark用の任意依存で、逐次教材には不要です。

GPUはCUDA Toolkitと対応GPU／ドライバ、OpenACCはさらにNVHPC nvc++が必要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=cufft
CPU_TARGET=fft_cpu
GPU_TARGET=fft_gpu
DIRECTIVE_TARGET=openacc_cufft
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## 正本ソースとターゲット

| 用途 | ソース | CMakeターゲット／実行ファイル |
| --- | --- | --- |
| CPU教材 | `examples/fft_cpu.c` | `fft_cpu` |
| CUDA教材 | `examples/fft_gpu.cu` | `fft_gpu` |
| OpenACC教材 | `examples/openacc_cufft.cpp` | `openacc_cufft` |
| CPUベンチマーク | `benchmarks/fft_cpu_bench.c` | `fft_cpu_bench` |
| CUDAベンチマーク | `benchmarks/fft_gpu_bench.cu` | `fft_gpu_bench` |
| OpenACCベンチマーク | `benchmarks/openacc_cufft_bench.cpp` | `openacc_cufft_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

精度・変換方向・stride・distance・batchを実際のbufferに合わせます。planは形状と配置が合う間だけ再利用します。FFTW教材は逐次の`FFTW_ESTIMATE`で、スレッド版benchmarkとは別です。

CUDA版はdevice配列を直接管理し、OpenACC版はdata領域と`host_data use_device`を使って同じGPUライブラリへ渡します。配列の所有権を混ぜず、結果を読む前に完了を保証してください。

<a id="direct-compile"></a>

## 直接コンパイル

CMakeを使わない場合の代替手順です。リポジトリ直下のBashで、必要なheader・libraryの検索パスが設定済みであることを前提とします。各方式のblockを選び、成功した場合だけ実行します。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 nvidia/c-cpp/cufft/examples/fft_cpu.c -lfftw3f -lm -o "$DIRECT_BUILD/fft_cpu" &&
"$DIRECT_BUILD/fft_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/cufft/examples/fft_gpu.cu -lcufft \
  -o "$DIRECT_BUILD/fft_gpu" &&
"$DIRECT_BUILD/fft_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/cufft/examples/openacc_cufft.cpp -cudalib=cufft \
  -o "$DIRECT_BUILD/openacc_cufft" &&
"$DIRECT_BUILD/openacc_cufft"
```

<a id="cmake-configure-and-build"></a>

## CMakeの構成とビルド

上の選択値を[共通手順](../../../docs/BUILDING.ja.md)へ渡します。CPU、CUDA、OpenACCの新しいtreeを用い、選択したtargetだけを構築します。CPU数値ライブラリが不要ならGPUの節だけを利用できます。

<a id="first-build-and-run"></a>

## 最初のビルドと実行

選択・build・実行の順に進み、終了値0と「教材の既定問題」の期待値を確認してください。欠落targetや非ゼロ終了を成功とは扱いません。共有計算機では計算ノードの割当内で実行します。

<a id="run-the-examples"></a>
Fortran教材は[別の案内](../../fortran/cufft/README.ja.md)を使います。

<a id="from-example-to-benchmark"></a>

## サンプルからベンチマークへの対応

教材に時間測定用の引数はありません。性能測定では対応する`_bench`を使い、library（内部scope名`compute`）とend-to-endを区別します。以下は既存benchmarkの参照情報です。[測定手順](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)でbenchmarkを有効にしたbuildを作り、そこで設定したパスを使ってください。最小教材buildにはbenchmarkは含まれません。

<a id="benchmark-cli"></a>

## ベンチマークCLI

以下はリンク先の測定手順で作ったbuildを使う1スレッド要求の小規模確認です。BLAS／Sparse／Solverはその手順のoneMKL構成に対応します。別providerでbuildした場合は`--cpu-backend`もbuildに合わせて変更してください。

threaded CPUの単独smoke実行例です。

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/cufft/fft_cpu_bench" \
  --size 256 --batch 8 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-4 \
  --rel-tolerance 1e-5 --output - --format jsonl \
  --cpu-backend cpu-fftw-threaded --cpu-threads 1 \
  --cpu-threads-effective 1 --cpu-backend-role production \
  --series-role primary --cpu-parallelism threaded
```

productionスイート実行では、すべての回数、role、検証閾値をeffectiveな正本設定から取得し、`tools/run_suite.py`だけがノードrawファイルを書き込みます。

<a id="timing-scopes"></a>

## 測定スコープ

表示名libraryの内部scopeは`compute`です。

`compute`は、プラン作成、メモリ確保、転送／状態復元、測定前の同期を済ませた後、変換の反復だけを測定します。`end-to-end`は各repeatのメモリ確保、プラン作成、入力転送、変換、完了、出力転送を測定し、後処理は終了時刻の後に行います。各repeat前に正本状態を復元します。warm-upと検証は測定区間外です。

<a id="verification"></a>

## 検証

effectiveな絶対・相対許容誤差を使い、出力を正本の解析的変換結果と比較します。検証失敗はその試行を失敗として記録しますが、状態を復元した後の試行の実行は妨げません。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCが入力・出力配列のメモリを所有します。`host_data use_device`は、すでに存在するデバイスアドレスをcuFFTへ渡します。ターゲットは専用のビルドツリーで`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::cufft`をリンクします。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

測定では選択したproduction CPU backendを明示し、依存不足による別providerへの置換を行いません。スレッド要求と実効worker数は区別します。FFTWの教材は逐次、スレッド版はbenchmarkの別系列です。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

PegasusでNVIDIA C/C++のbuild・数値smoke・性能測定の保存記録があります。[版と確認範囲](../../../docs/VALIDATION_REPORT.md#current-coexistence-status)を参照してください。ローカルfixture検査と実機検証は別で、別環境や後続変更の動作を保証しません。数値誤差・乱数分布の検証条件とCPU並列性の制約は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。
