# cuRAND C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

FP64の一様乱数を生成します。シミュレーション入力や確率的アルゴリズムで使うgeneratorと出力bufferの扱いを学べます。

<a id="teaching-default"></a>

## 教材の既定問題

seed 1234で`2^24`個を生成します。CPUは逐次`std::mt19937_64`、GPUはライブラリ既定の疑似乱数generatorです。CPUの区間は[0,1)、GPUは(0,1]です。同じ分布の比較であり、同じ乱数列ではありません。
同じ分布・出力型の比較であり、同じ乱数アルゴリズムの比較ではありません。

<a id="dependencies"></a>

## 依存環境

CPUはC++17と標準ライブラリだけで動きます。GPUにはcuRANDが必要です。

GPUはCUDA Toolkitと対応GPU／ドライバ、OpenACCはさらにNVHPC nvc++が必要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=c-cpp
LIBRARY=curand
CPU_TARGET=rand_cpu
GPU_TARGET=rand_gpu
DIRECTIVE_TARGET=openacc_curand
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## 正本ソースとターゲット

| 用途 | ソース | CMakeターゲット／実行ファイル |
| --- | --- | --- |
| CPU教材 | `examples/rand_cpu.cpp` | `rand_cpu` |
| CUDA教材 | `examples/rand_gpu.cu` | `rand_gpu` |
| OpenACC教材 | `examples/openacc_curand.cpp` | `openacc_curand` |
| CPUベンチマーク | `benchmarks/rand_cpu_bench.cpp` | `rand_cpu_bench` |
| CUDAベンチマーク | `benchmarks/rand_gpu_bench.cu` | `rand_gpu_bench` |
| OpenACCベンチマーク | `benchmarks/openacc_curand_bench.cpp` | `openacc_curand_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

用途に合うgeneratorとseed／streamの方針を選びます。同じ列の再生成を意図しない限り、毎回seedを戻しません。継続する列ではgeneratorとbufferを再利用します。CPUとGPUのアルゴリズムは異なるため、要素一致ではなく範囲・分布統計で確認します。CPU教材は逐次です。

CUDA版はdevice配列を直接管理し、OpenACC版はdata領域と`host_data use_device`を使って同じGPUライブラリへ渡します。配列の所有権を混ぜず、結果を読む前に完了を保証してください。

<a id="direct-compile"></a>

## 直接コンパイル

CMakeを使わない場合の代替手順です。リポジトリ直下のBashで、必要なheader・libraryの検索パスが設定済みであることを前提とします。各方式のblockを選び、成功した場合だけ実行します。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 nvidia/c-cpp/curand/examples/rand_cpu.cpp -o "$DIRECT_BUILD/rand_cpu" &&
"$DIRECT_BUILD/rand_cpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVCC_ARCH:?Set the nvcc architecture supported by your GPU}"
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O2 -arch="$NVCC_ARCH" \
  nvidia/c-cpp/curand/examples/rand_gpu.cu -lcurand \
  -o "$DIRECT_BUILD/rand_gpu" &&
"$DIRECT_BUILD/rand_gpu"
```

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set your NVHPC GPU target}"
NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT" \
nvc++ -std=c++17 -O2 -acc -cuda -gpu="$NVHPC_GPU_TARGET" \
  nvidia/c-cpp/curand/examples/openacc_curand.cpp -cudalib=curand \
  -o "$DIRECT_BUILD/openacc_curand" &&
"$DIRECT_BUILD/openacc_curand"
```

<a id="cmake-configure-and-build"></a>

## CMakeの構成とビルド

上の選択値を[共通手順](../../../docs/BUILDING.ja.md)へ渡します。CPU、CUDA、OpenACCの新しいtreeを用い、選択したtargetだけを構築します。CPU数値ライブラリが不要ならGPUの節だけを利用できます。

<a id="first-build-and-run"></a>

## 最初のビルドと実行

選択・build・実行の順に進み、終了値0と「教材の既定問題」の期待値を確認してください。欠落targetや非ゼロ終了を成功とは扱いません。共有計算機では計算ノードの割当内で実行します。

<a id="run-the-examples"></a>
Fortran教材は[別の案内](../../fortran/curand/README.ja.md)を使います。

<a id="from-example-to-benchmark"></a>

## サンプルからベンチマークへの対応

教材に時間測定用の引数はありません。性能測定では対応する`_bench`を使い、library（内部scope名`compute`）とend-to-endを区別します。以下は既存benchmarkの参照情報です。[測定手順](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)でbenchmarkを有効にしたbuildを作り、そこで設定したパスを使ってください。最小教材buildにはbenchmarkは含まれません。

<a id="benchmark-cli"></a>

## ベンチマークCLI

以下はリンク先の測定手順で作ったbuildを使う1スレッド要求の小規模確認です。BLAS／Sparse／Solverはその手順のoneMKL構成に対応します。別providerでbuildした場合は`--cpu-backend`もbuildに合わせて変更してください。

```bash
"$CPU_CUDA_BUILD/nvidia/c-cpp/curand/rand_cpu_bench" \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --sigma-multiplier 6 --expected-mean 0.5 \
  --expected-second-central-moment 0.08333333333333333 \
  --output - --format jsonl --cpu-backend cpu-std-random-serial \
  --cpu-threads 1 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

productionスイート実行は、統計定数をすべて実効設定から渡し、ランナーだけがノードrawファイルを書き込みます。

<a id="timing-scopes"></a>

## 測定スコープ

表示名libraryの内部scopeは`compute`です。

`compute`は、測定前に永続生成器を作成してシードを設定し、設定されたオフセットを復元してから、既存メモリへの生成の反復を測定します。`end-to-end`は各repeatのメモリと生成器の作成、設定、生成、完了、ホストでの結果取得を含み、破棄は終了時刻の後に行います。正本の生成器状態は、スコープが要求する各raw試行／repeatの前に復元します。

<a id="verification"></a>

## 検証

両方の乱数区間に、端点を含む共通の範囲確認`0 <= x <= 1`を適用します。メタデータにはCPUの`[0,1)`とcuRANDの`(0,1]`という契約を記録します。必須指標は`observed_min`、`observed_max`、`sample_mean`、`second_central_moment_about_half`です。設定上の許容幅は、平均0.5に対して`sigma_multiplier*sqrt(1/(12*N))`、第2中心モーメント1/12に対して`sigma_multiplier*sqrt(1/(180*N))`です。

<a id="openacc-notes"></a>

## OpenACCの注意点

データ領域が出力メモリを所有し、`host_data use_device`でcuRANDへ渡します。ターゲットは`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::curand`をリンクします。移植可能なソースにGPUアーキテクチャは固定しません。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

保存済みCPU評価は逐次です。要求スレッド数が多くても並列版にはなりません。自分のマシンのCPU名・選択したbackend・要求値と実効値を区別して記録します。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

PegasusでNVIDIA C/C++のbuild・数値smoke・性能測定の保存記録があります。[版と確認範囲](../../../docs/VALIDATION_REPORT.md#current-coexistence-status)を参照してください。ローカルfixture検査と実機検証は別で、別環境や後続変更の動作を保証しません。数値誤差・乱数分布の検証条件とCPU並列性の制約は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。
