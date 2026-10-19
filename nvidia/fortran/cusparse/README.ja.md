# cuSPARSE — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

CSR形式のFP64疎行列ベクトル積`y = alpha*A*x + beta*y`を計算します。教材行列は規則的な2次元5点Poisson stencilです。

<a id="problem"></a>

## 問題と期待する結果

1024×1024格子から1,048,576行の行列を生成します。対角4、隣接−1、`alpha=beta=1`、`x=y=1`です。出力yは内部1・辺2・角3となります。CSR添字は1始まりです。

Fortran教材は終了値0と`verification PASS`を確認します。固定サイズはソース内定数です。

<a id="dependencies"></a>

## 必要環境

CPUはoneMKL Sparse LP64が必要です。`MKLROOT`を設定し、`mkl_rt`・`mkl_service.h`・`mkl_spblas.f90`を同じインストールから指定します。別compilerで作った`.mod`を流用しません。

CPUはGNU Fortran／NVHPC、GPUはNVHPC `nvfortran`とCUDA Toolkitを使います。Fortranの`.mod`はcompiler間で共有せず、C/C++と別treeを使います。[Fortran共通案内](../README.ja.md)にABIと数値条件をまとめています。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cusparse
CPU_TARGET=sparse_cpu
GPU_TARGET=sparse_gpu
DIRECTIVE_TARGET=openacc_cusparse
CPU_OPTIONS=()
```

<a id="sources"></a>

## ソースと明示リンク入力

| 方式 | 教材 | benchmark |
| --- | --- | --- |
| CPU | [sparse_cpu.f90](examples/sparse_cpu.f90) / `sparse_cpu` | [sparse_cpu_bench.f90](benchmarks/sparse_cpu_bench.f90) / `sparse_cpu_bench` |
| CUDA | [sparse_gpu.f90](examples/sparse_gpu.f90) / `sparse_gpu` | [sparse_gpu_bench.f90](benchmarks/sparse_gpu_bench.f90) / `sparse_gpu_bench` |
| OpenACC | [openacc_cusparse.f90](examples/openacc_cusparse.f90) / `openacc_cusparse` | [openacc_cusparse_bench.f90](benchmarks/openacc_cusparse_bench.f90) / `openacc_cusparse_bench` |

3方式とも行列生成helper [make_poisson2d_csr.f90](../../../common/fortran/make_poisson2d_csr.f90)をリンクします。CMakeは自動的に含めます。

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

CSRのrow pointer・column index・値を一組として置き換え、添字幅・基準・次元・非零要素の順序を一致させます。betaが0でなければ初期yも必要です。descriptorとanalysisは保持する行列・pointerについてAPIが許す範囲で再利用します。規則的な5点行列の性能を不規則な実行列へ一般化しないでください。

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
  nvidia/fortran/cusparse/examples/sparse_cpu.f90 common/fortran/make_poisson2d_csr.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/sparse_cpu" &&
"$DIRECT_BUILD/sparse_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/sparse_gpu.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/sparse_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/openacc_cusparse.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/openacc_cusparse" &&
"$DIRECT_BUILD/sparse_gpu" &&
"$DIRECT_BUILD/openacc_cusparse"
```

<a id="examples-and-a-small-benchmark"></a>

## 教材の次に：必要なら性能測定

最小buildにはbenchmarkは含まれません。性能を測る場合は[Fortran測定手順](../README.ja.md#first-checks-and-own-measurements)でbenchmarkを有効化し、言語別configとmanifestを使います。libraryの内部CLI名は`compute`、もう一方は`end-to-end`です。教材の`main`全体とは測定範囲が異なります。

数学と計測条件は[問題仕様](../../../docs/PROJECT_SPECIFICATION.md)・[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。Fortranの結果をC/C++結果へ混ぜず、CPU組込み乱数・sumは逐次系列として扱います。

<a id="validation"></a>

## 確認済みの範囲

PegasusでNVHPCによるbuild・数値smoke・本測定の記録があります。[版と制約](../../../docs/VALIDATION_REPORT.md#fortran-production-validation)を参照してください。別のcompiler版やGPUでの動作保証ではありません。CPU要求スレッド数と実効数、リンクしたlibraryと完全な実ロード来歴は区別します。
