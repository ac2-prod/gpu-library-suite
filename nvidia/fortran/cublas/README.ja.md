# cuBLAS — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

FP64 DGEMMで密行列積`C = alpha*A*B + beta*C`を計算します。CPU BLASからGPU直接呼び出し・ディレクティブによる配列管理への対応を確認できます。

<a id="problem"></a>

## 問題と期待する結果

列優先の正方行列で`m=n=k=1024`、`alpha=beta=1`、A/B/Cはすべて1です。出力は全要素`1025`となります。

Fortran教材は終了値0と`verification PASS`を確認します。固定サイズはソース内定数です。

<a id="dependencies"></a>

## 必要環境

CPUはoneMKL LP64を使います。`MKLROOT`を設定し、`mkl_rt`と`mkl_service.h`を同じインストールから指定します。

CPUはGNU Fortran／NVHPC、GPUはNVHPC `nvfortran`とCUDA Toolkitを使います。Fortranの`.mod`はcompiler間で共有せず、C/C++と別treeを使います。[Fortran共通案内](../README.ja.md)にABIと数値条件をまとめています。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cublas
CPU_TARGET=blas_cpu
GPU_TARGET=blas_gpu
DIRECTIVE_TARGET=openacc_cublas
CPU_OPTIONS=()
```

<a id="sources"></a>

## ソースと明示リンク入力

| 方式 | 教材 | benchmark |
| --- | --- | --- |
| CPU | [blas_cpu.f90](examples/blas_cpu.f90) / `blas_cpu` | [blas_cpu_bench.f90](benchmarks/blas_cpu_bench.f90) / `blas_cpu_bench` |
| CUDA | [blas_gpu.f90](examples/blas_gpu.f90) / `blas_gpu` | [blas_gpu_bench.f90](benchmarks/blas_gpu_bench.f90) / `blas_gpu_bench` |
| OpenACC | [openacc_cublas.f90](examples/openacc_cublas.f90) / `openacc_cublas` | [openacc_cublas_bench.f90](benchmarks/openacc_cublas_bench.f90) / `openacc_cublas_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

配列の格納順とleading dimension（`lda`、`ldb`、`ldc`）を合わせます。betaが0でないためCも入力です。A/Bだけをコピーしてはいけません。繰り返す場合は配列とhandleを適切な寿命で保持します。CPUの並列性は選択したBLAS providerが担い、要求スレッド数は実測worker数ではありません。

Fortranの列優先配列とC ABIの型を区別します。CUDA Fortranはdevice配列、OpenACCはdata領域を使い、同じGPUライブラリへ渡します。helperやwrapperもアプリケーションのリンク入力へ含めてください。

<a id="direct-compilation"></a>

## 手動コンパイル（NVHPCでの代替手順）

通常は共通CMake手順を使います。以下はリポジトリ直下のBashで、必要な探索パスが設定済みの場合の手動手順です。CPUとGPUは別blockなので、必要な方式だけを使えます。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cublas/examples/blas_cpu.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/blas_cpu" &&
"$DIRECT_BUILD/blas_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cublas \
  nvidia/fortran/cublas/examples/blas_gpu.f90 -o "$DIRECT_BUILD/blas_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cublas \
  nvidia/fortran/cublas/examples/openacc_cublas.f90 -o "$DIRECT_BUILD/openacc_cublas" &&
"$DIRECT_BUILD/blas_gpu" &&
"$DIRECT_BUILD/openacc_cublas"
```

<a id="examples-and-a-small-benchmark"></a>

## 教材の次に：必要なら性能測定

最小buildにはbenchmarkは含まれません。性能を測る場合は[Fortran測定手順](../README.ja.md#first-checks-and-own-measurements)でbenchmarkを有効化し、言語別configとmanifestを使います。libraryの内部CLI名は`compute`、もう一方は`end-to-end`です。教材の`main`全体とは測定範囲が異なります。

数学と計測条件は[問題仕様](../../../docs/PROJECT_SPECIFICATION.md)・[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。Fortranの結果をC/C++結果へ混ぜず、CPU組込み乱数・sumは逐次系列として扱います。

<a id="validation"></a>

## 確認済みの範囲

PegasusでNVHPCによるbuild・数値smoke・本測定の記録があります。[版と制約](../../../docs/VALIDATION_REPORT.md#fortran-production-validation)を参照してください。別のcompiler版やGPUでの動作保証ではありません。CPU要求スレッド数と実効数、リンクしたlibraryと完全な実ロード来歴は区別します。
