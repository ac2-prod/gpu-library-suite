# Thrust C/C++のサンプルとベンチマーク

[English](README.md) | [日本語](README.ja.md)

このディレクトリでは、FP64二乗和reductionの正本問題を、C++17 STL、直接Thrust、OpenACC/Thrust連携で比較します。ワークロードの定義は[`docs/PROJECT_SPECIFICATION.md`（英語）](../../../docs/PROJECT_SPECIFICATION.md)、CLI、時間測定、状態復元、検証は[`docs/BENCHMARK_PROTOCOL.ja.md`](../../../docs/BENCHMARK_PROTOCOL.ja.md)が定めます。

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

<a id="teaching-default"></a>

## 教材の既定問題

各教材プログラムは`2^24`個のFP64値を作成し、transform-reduceの流れで`sum(values[i]^2)`を評価します。ベンチマーク基盤を含まない、直接実行できる単一ソースのサンプルです。

<a id="dependencies"></a>

## 依存環境

逐次CPUターゲットに必要なのはC++17標準ライブラリだけです。明示的に選択するOpenMP CPU版には`OpenMP::OpenMP_CXX`が必要です。CUDAターゲットにはCUDA RuntimeとThrustヘッダが必要です。OpenACCターゲットにはNVHPC OpenACC C++、CUDA Runtime、Thrustに加え、コンパイル時とリンク時の両方に`-cuda`相当のフラグが必要です。CMakeは`CUDAToolkit_INCLUDE_DIRS`から`include/cccl/thrust/version.h`を解決し、CUDA/OpenACCの検査、サンプル、ベンチマークのすべてについて、その外部Toolkitの`include/cccl`を検索順の先頭に置きます。

<a id="direct-compile"></a>

## 直接コンパイル

```bash
mkdir -p /tmp/gpu-library-suite-local-build
c++ -std=c++17 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o /tmp/gpu-library-suite-local-build/reduce_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/thrust/examples/reduce_gpu.cu \
  -o /tmp/gpu-library-suite-local-build/reduce_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/thrust/examples/openacc_thrust.cpp \
  -o /tmp/gpu-library-suite-local-build/openacc_thrust-direct
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
  --target reduce_cpu reduce_gpu reduce_cpu_bench reduce_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_thrust openacc_thrust_bench
```

<a id="run-the-examples"></a>

## サンプルの実行

上記の対応ターゲットをビルドした後、リポジトリのルートから実行します。

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/thrust/reduce_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/thrust/reduce_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/thrust/openacc_thrust"
printf 'OpenACC exit status: %s\n' "$?"
```

二乗和の結果を全要素1の入力要素数と比較し、各終了ステータスを確認してください。ターゲットがない場合や、終了値が0以外の場合、比較は未完了です。続いて[測定と結果処理](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)へ進んでください。

<a id="benchmark-cli"></a>

## ベンチマークCLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/thrust/reduce_cpu_bench \
  --size 65536 --warmup 1 --repeat 2 --trials 1 --scope compute \
  --verify true --abs-tolerance 1e-12 --rel-tolerance 1e-10 \
  --output - --format jsonl --cpu-backend cpu-stl-serial \
  --cpu-threads 48 --cpu-threads-effective 1 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism serial
```

production実行は閾値とroleを実効設定から取得し、`tools/run_suite.py`だけがノードraw-resultファイルを書き込みます。

<a id="timing-scopes"></a>

## 測定スコープ

`compute`は、測定前に入力／デバイスメモリを準備し、transform-reduceの反復だけを測定します。GPUの`end-to-end`は各repeatのデバイスメモリ作成、入力転送、reduction、完了／結果取得を含み、後処理は終了時刻の後に行います。CPUベクトルと正本の全要素1のホスト入力は測定区間外で準備し、CPU経路にはデバイス転送はありません。正本状態の復元と検証も測定区間外です。

<a id="verification"></a>

## 検証

測定結果を正本の解析的な二乗和と、`abs_tolerance + rel_tolerance * reference_scale`を使って比較します。検証失敗は保持しますが、状態復元後の試行を妨げる致命的失敗とは扱いません。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

`cpu-stl-serial`は、設定されたproductionのprimary逐次CPUバックエンドです。承認済み凡例では、並列の基準実装としてではなく、**Intel Xeon Platinum 8468, STL (single thread)**と表示します。要求スレッド数が48でも、有効スレッド数は1です。この系列からspeedupは描画しません。任意の`cpu-openmp`ビルドは別名のバックエンドであり、暗黙の代替にはなりません。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCが入力メモリの確保と転送を所有し、その後`thrust::device_pointer_cast`で既知のデバイスアドレスを変換します。Thrustの同期アルゴリズムは、通常ホストで結果を取得するための追加の完了待ちを必要としません。教材のOpenACCサンプルでは、連携の境界を示すために明示的な`cudaDeviceSynchronize`を残しています。コンパイルとリンクの両方で、NVHPC CUDA C++連携フラグを指定します。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

reductionの順序は実装間で異なることがあるため、検証ではビット単位の一致ではなく、宣言された浮動小数点許容誤差を使います。ローカルテストは逐次CPU実装と模擬ヘッダ GPU構文を対象とします。実Thrust/CUDA、NVHPC、GPU、OpenMPのproduction設定、Pegasus実行はローカルでは未検証です。その後に保存された実機実行記録は、[検証報告（英語）](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)でこれらのローカルテストと区別しています。
