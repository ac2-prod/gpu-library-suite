# cuSPARSE C/C++のサンプルとベンチマーク

[English](README.md) | [日本語](README.ja.md)

このディレクトリでは、FP64の5点Poisson CSR SpMVの正本問題を、oneMKL Sparse、直接cuSPARSE、OpenACC/cuSPARSEで比較します。行列の構成は[`docs/PROJECT_SPECIFICATION.md`（英語）](../../../docs/PROJECT_SPECIFICATION.md)、CLI、時間測定、状態復元、検証は[`docs/BENCHMARK_PROTOCOL.ja.md`](../../../docs/BENCHMARK_PROTOCOL.ja.md)が定めます。

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

<a id="teaching-default"></a>

## 教材の既定問題

教材プログラムは1,024×1,024の格子を構成するため、行列は1,048,576行、非零要素数は`5*n - 2*nx - 2*ny`です。`alpha=1`、`beta=1`でFP64 SpMVを実行します。正本のCPU教材サンプルはoneMKL Sparse専用で、手書きのreference実装へのフォールバックは含みません。

<a id="dependencies"></a>

## 依存環境

`sparse_cpu`にはoneMKL Sparseが必要で、記述子/create、hint/optimize、SpMV、destroyを対象とするコンパイル・リンク検査が成功しなければなりません。CPUベンチマークは、`GPU_SUITE_CPU_SPARSE_BACKEND=REFERENCE`の場合、別名の`cpu-reference-csr` バックエンドとしてビルドすることもできます。この選択では、正本のCPU教材ターゲットを明確な理由付きで無効化します。CUDA/OpenACCターゲットにはCUDA RuntimeとcuSPARSEの記述子/SpMVシンボルが必要です。

<a id="direct-compile"></a>

## 直接コンパイル

以下の代表的なコマンドは、選択したライブラリがコンパイラ検索パスに含まれることを前提とします。

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cusparse/examples/sparse_cpu.c \
  -lmkl_rt -lpthread -ldl -lm \
  -o /tmp/gpu-library-suite-local-build/sparse_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cusparse/examples/sparse_gpu.cu \
  -lcusparse -o /tmp/gpu-library-suite-local-build/sparse_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp -cudalib=cusparse \
  -o /tmp/gpu-library-suite-local-build/openacc_cusparse-direct
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
  -DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target sparse_cpu sparse_gpu sparse_cpu_bench sparse_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cusparse openacc_cusparse_bench
```

<a id="run-the-examples"></a>

## サンプルの実行

上記の対応ターゲットをビルドした後、リポジトリのルートから実行します。

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusparse/sparse_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cusparse/sparse_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cusparse/openacc_cusparse"
printf 'OpenACC exit status: %s\n' "$?"
```

Poisson SpMVの誤差と各終了ステータスを確認してください。ターゲットがない場合や、終了値が0以外の場合、3実装の確認は未完了です。続いて[測定と結果処理](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)へ進んでください。

<a id="benchmark-cli"></a>

## ベンチマークCLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cusparse/sparse_cpu_bench \
  --size 4096 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-onemkl --cpu-threads 48 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

平方数の`--size`から、`nx`、`ny`、厳密な非零要素数を決定します。スイート実行では設定が定めるroleと閾値を渡し、raw-resultファイルの書込み主体をランナーだけに限定します。

<a id="timing-scopes"></a>

## 測定スコープ

`compute`は、測定前に記述子とメモリを準備し、yを復元してから、repeatを通してSpMVの状態更新を連続適用します。検証は全repeatを反映した出力を確認します。`end-to-end`は、各repeatのメモリ、転送、記述子作成、作業領域の照会／確保、SpMV、完了、copyoutを測定します。後処理は終了時刻の後に行い、各repeatの前に正本のyを復元します。

<a id="verification"></a>

## 検証

最大絶対誤差を`abs_tolerance + rel_tolerance * reference_scale`と比較します。検証失敗は復旧可能な試行の失敗です。残りの試行をskipするのは、致命的な記述子、メモリ確保、API、clockの失敗だけです。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

`cpu-onemkl`がproductionのprimaryバックエンドです。`cpu-reference-csr`はベンチマーク用途だけの明示的なreferenceバックエンドであり、正本教材サンプル内のoneMKLを置き換えたり、黙ってproduction系列になったりすることはできません。

<a id="openacc-notes"></a>

## OpenACCの注意点

データ領域がCSRとベクトルのメモリを所有し、`host_data use_device`がそのアドレスをcuSPARSEへ渡します。CUDAで確保するのはサイズが0でないのcuSPARSE作業領域だけで、0バイトの作業領域は`nullptr`のままです。ターゲットは`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::cusparse`をリンクします。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

正本スイート設定が受け付ける問題サイズは平方数だけです。メモリ確保の失敗は、自動的に小さくせず報告します。ローカルテストはfake oneMKLプロバイダとfake CUDAヘッダを使います。production用oneMKL Sparseランタイム、実cuSPARSE、NVHPC、GPU、Pegasus実行はローカルでは未検証です。その後に保存された実機実行記録は、[検証報告（英語）](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)でこれらのローカルテストと区別しています。
