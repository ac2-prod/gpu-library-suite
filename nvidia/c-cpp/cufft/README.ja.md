# cuFFT C/C++のサンプルとベンチマーク

[English](README.md) | [日本語](README.ja.md)

このディレクトリでは、batched FP32順方向複素変換の正本問題を、FFTW3f、直接cuFFT、OpenACC/cuFFTで比較します。問題の数学的定義は[`docs/PROJECT_SPECIFICATION.md`（英語）](../../../docs/PROJECT_SPECIFICATION.md)、CLI、時間測定、状態復元、検証は[`docs/BENCHMARK_PROTOCOL.ja.md`](../../../docs/BENCHMARK_PROTOCOL.ja.md)が定めます。

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

<a id="teaching-default"></a>

## 教材の既定問題

3本の教材プログラムはすべて、長さ1,024の独立したC2C順方向変換を4,096本、単精度で実行します。完全な単一ソースプログラムであり、ベンチマークCLIや結果処理基盤は意図的に含めていません。

<a id="dependencies"></a>

## 依存環境

CPUプログラムにはFFTW3fが必要です。threadedベンチマークには、さらにFFTW3f threadsライブラリと、その初期化・スレッド数指定・後処理のシンボルが必要です。CUDAプログラムにはCUDA RuntimeとcuFFT、OpenACCプログラムにはNVHPC OpenACC C++、CUDA Runtime、cuFFTが必要です。CMakeはコンパイル・リンク検査を使い、影響するターゲットを無効化した理由を報告します。

<a id="direct-compile"></a>

## 直接コンパイル

リポジトリのルートで、ローカルインストール環境に合わせて通常のコンパイラ検索パスを調整してください。

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cufft/examples/fft_cpu.c \
  -lfftw3f -lm -o /tmp/gpu-library-suite-local-build/fft_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cufft/examples/fft_gpu.cu \
  -lcufft -o /tmp/gpu-library-suite-local-build/fft_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cufft/examples/openacc_cufft.cpp -cudalib=cufft \
  -o /tmp/gpu-library-suite-local-build/openacc_cufft-direct
```

<a id="cmake-configure-and-build"></a>

## CMakeのconfigureとビルド

CPUとCUDAは1つのツリーに、OpenACCは別のNVHPCツリーに配置します。`CUDA_TOOLKIT_ROOT`、`CUDA_ARCHITECTURES`、`NVHPC_GPU_TARGET`に、サイトで承認された具体値が設定済みであることを前提とします。

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target fft_cpu fft_gpu fft_cpu_bench fft_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cufft openacc_cufft_bench
```

CUDAアーキテクチャ、Toolkitルート、NVHPC GPUターゲットは明示的な外部入力です。移植可能なソース自身が選択することはありません。

<a id="run-the-examples"></a>

## サンプルの実行

上記の対応ターゲットをビルドした後、リポジトリのルートから実行します。

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cufft/fft_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cufft/fft_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cufft/openacc_cufft"
printf 'OpenACC exit status: %s\n' "$?"
```

報告される変換誤差を確認し、利用可能な各サンプルが0で終了することを確認してください。ターゲットがない場合や終了値が0以外の場合を、比較に成功したものと扱わないでください。CPU教材プログラムは逐次FFTWを使い、threadedベンチマークバックエンドではありません。続いて[測定と結果処理](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)へ進んでください。

<a id="benchmark-cli"></a>

## ベンチマークCLI

threaded CPUの単独smoke実行例です。

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cufft/fft_cpu_bench \
  --size 256 --batch 8 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-4 \
  --rel-tolerance 1e-5 --output - --format jsonl \
  --cpu-backend cpu-fftw-threaded --cpu-threads 48 \
  --cpu-threads-effective 48 --cpu-backend-role production \
  --series-role primary --cpu-parallelism threaded
```

productionスイート実行では、すべての回数、role、検証閾値をeffectiveな正本設定から取得し、`tools/run_suite.py`だけがノードrawファイルを書き込みます。

<a id="timing-scopes"></a>

## 測定スコープ

`compute`は、プラン作成、メモリ確保、転送／状態復元、測定前の同期を済ませた後、変換の反復だけを測定します。`end-to-end`は各repeatのメモリ確保、プラン作成、入力転送、変換、完了、出力転送を測定し、後処理は終了時刻の後に行います。各repeat前に正本状態を復元します。warm-upと検証は測定区間外です。

<a id="verification"></a>

## 検証

effectiveな絶対・相対許容誤差を使い、出力を正本の解析的変換結果と比較します。検証失敗はその試行を失敗として記録しますが、状態を復元した後の試行の実行は妨げません。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

Pegasus教材掲載用設定のCPU系列は`cpu-fftw-threaded`だけで、承認済み図では**Intel Xeon Platinum 8468, FFTW (48 C)**と表示します。FFTW threads APIを使い、保存済み教材掲載用ライブラリはpthread版の`--enable-threads`でビルドされています。CMakeは教材との対応確認用として、別途呼び出す`cpu-fftw-serial` バックエンドを保持できますが、教材掲載用系列ではなく、threaded FFTWの代わりに使うこともありません。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCが入力・出力配列のメモリを所有します。`host_data use_device`は、すでに存在するデバイスアドレスをcuFFTへ渡します。ターゲットは専用のビルドツリーで`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::cufft`をリンクします。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

batchと長さの組合せが非常に大きい場合、メモリ確保に失敗することがあります。その場合は、黙ってサイズを変更せず失敗として報告します。ローカルCPUテストと模擬ヘッダ構文テストでは、実CUDA GPU、cuFFTランタイム、NVHPCコンパイラ、Pegasus実行を確認していません。これらのproduction依存環境はローカルでは未検証であり、[`docs/PEGASUS_EXECUTION.md`（英語）](../../../docs/PEGASUS_EXECUTION.md)に記載する手動Pegasus確認が必要です。その後に保存された実機実行記録は、[検証報告（英語）](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)でこれらのローカルテストと区別しています。
