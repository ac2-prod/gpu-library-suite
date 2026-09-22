# cuRAND C/C++のサンプルとベンチマーク

[English](README.md) | [日本語](README.ja.md)

このディレクトリでは、C++17の`std::mt19937_64`、直接cuRAND、OpenACC/cuRANDによる一様倍精度乱数生成を比較します。ワークロードの定義は[`docs/PROJECT_SPECIFICATION.md`（英語）](../../../docs/PROJECT_SPECIFICATION.md)、CLI、時間測定、統計的検証は[`docs/BENCHMARK_PROTOCOL.ja.md`](../../../docs/BENCHMARK_PROTOCOL.ja.md)が定めます。

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

<a id="teaching-default"></a>

## 教材の既定問題

各教材プログラムは`2^24`個のFP64一様乱数を生成します。CPUは`std::mt19937_64`、GPU版は`CURAND_RNG_PSEUDO_DEFAULT`を使います。これは同じ分布／出力型のタスクの比較であり、同一の乱数アルゴリズムの比較ではありません。したがって、CPU/GPU間の要素ごとの一致は想定せず、検証もしません。

<a id="dependencies"></a>

## 依存環境

CPUターゲットに必要なのはC++17標準ライブラリだけです。CUDAターゲットにはCUDA RuntimeとcuRAND生成器 APIが必要です。OpenACCターゲットにはNVHPC OpenACC C++、CUDA Runtime、cuRANDが必要です。CMakeのコンパイル・リンク検査で生成器作成と一様倍精度乱数生成を確認した後、GPUターゲットを有効化します。

<a id="direct-compile"></a>

## 直接コンパイル

```bash
mkdir -p /tmp/gpu-library-suite-local-build
c++ -std=c++17 nvidia/c-cpp/curand/examples/rand_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/rand_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/curand/examples/rand_gpu.cu \
  -lcurand -o /tmp/gpu-library-suite-local-build/rand_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/curand/examples/openacc_curand.cpp -cudalib=curand \
  -o /tmp/gpu-library-suite-local-build/openacc_curand-direct
```

<a id="cmake-configure-and-build"></a>

## CMakeのconfigureとビルド

`CUDA_TOOLKIT_ROOT`、`CUDA_ARCHITECTURES`、`NVHPC_GPU_TARGET`に、サイトで承認された具体値が設定済みであることを前提とします。

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target rand_cpu rand_gpu rand_cpu_bench rand_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_curand openacc_curand_bench
```

<a id="run-the-examples"></a>

## サンプルの実行

上記の対応ターゲットをビルドした後、リポジトリのルートから実行します。

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/curand/rand_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/curand/rand_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/curand/openacc_curand"
printf 'OpenACC exit status: %s\n' "$?"
```

各終了ステータスと報告される分布の統計量を確認してください。RNGアルゴリズムが異なるため、CPU/GPUの要素ごとの一致は期待しないでください。ターゲットがない場合や終了値が0以外の場合を、3実装の確認に成功したものとは扱いません。続いて[測定と結果処理](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)へ進んでください。ベンチマークの検証では、以下の明示的な統計閾値を使います。

<a id="benchmark-cli"></a>

## ベンチマークCLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/curand/rand_cpu_bench \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --sigma-multiplier 6 --expected-mean 0.5 \
  --expected-second-central-moment 0.08333333333333333 \
  --output - --format jsonl --cpu-backend cpu-std-random-serial \
  --cpu-threads 48 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

productionスイート実行は、統計定数をすべて実効設定から渡し、ランナーだけがノードrawファイルを書き込みます。

<a id="timing-scopes"></a>

## 測定スコープ

`compute`は、測定前に永続生成器を作成してシードを設定し、設定されたオフセットを復元してから、既存メモリへの生成の反復を測定します。`end-to-end`は各repeatのメモリと生成器の作成、設定、生成、完了、ホストでの結果取得を含み、破棄は終了時刻の後に行います。正本の生成器状態は、スコープが要求する各raw試行／repeatの前に復元します。

<a id="verification"></a>

## 検証

両方の乱数区間に、端点を含む共通の範囲確認`0 <= x <= 1`を適用します。メタデータにはCPUの`[0,1)`とcuRANDの`(0,1]`という契約を記録します。必須指標は`observed_min`、`observed_max`、`sample_mean`、`second_central_moment_about_half`です。設定上の許容幅は、平均0.5に対して`sigma_multiplier*sqrt(1/(12*N))`、第2中心モーメント1/12に対して`sigma_multiplier*sqrt(1/(180*N))`です。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

`cpu-std-random-serial`は、設定されたproductionのprimary逐次CPU実装です。承認済み凡例は**Intel Xeon Platinum 8468, std::mt19937_64 (single thread)**であり、並列または同一アルゴリズムの基準実装ではありません。キャンペーンで48スレッドを要求していても、有効スレッド数は1のままです。この系列からspeedupは描画しません。

<a id="openacc-notes"></a>

## OpenACCの注意点

データ領域が出力メモリを所有し、`host_data use_device`でcuRANDへ渡します。ターゲットは`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::curand`をリンクします。移植可能なソースにGPUアーキテクチャは固定しません。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

統計的テストは系列の同一性ではなく分布の振る舞いを比較します。有限標本は、設定された許容幅によって失敗することがあります。ローカルテストで実行するのは逐次CPU生成器とfake GPU構文だけです。実cuRANDランタイム、NVHPCコンパイラ、GPU、Pegasusキャンペーンはローカルでは未検証です。その後に保存された実機実行記録は、[検証報告（英語）](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)でこれらのローカルテストと区別しています。
