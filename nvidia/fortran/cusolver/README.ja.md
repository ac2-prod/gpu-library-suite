# cuSOLVER — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

LU分解を使って複数右辺のFP64密行列方程式`A*X=B`を解きます。行列のほかpivot配列とsolverの終了情報を扱う流れを示します。

<a id="problem"></a>

## 問題と期待する結果

`N=1024`、`nrhs=16`、`A=N*I+ones`、Bの全要素は`2*N`です。解はすべて1となります。APIの成功と`info=0`に加え、数値検証の成功が必要です。

Fortran教材は終了値0と`verification PASS`を確認します。固定サイズはソース内定数です。

<a id="dependencies"></a>

## 必要環境

CPUはoneMKL LP64のFortran LAPACKを使います。`MKLROOT`を設定し、`mkl_rt`と`mkl_service.h`を同じインストールから指定します。C/C++用LAPACKEは不要です。

CPUはGNU Fortran／NVHPC、GPUはNVHPC `nvfortran`とCUDA Toolkitを使います。Fortranの`.mod`はcompiler間で共有せず、C/C++と別treeを使います。[Fortran共通案内](../README.ja.md)にABIと数値条件をまとめています。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cusolver
CPU_TARGET=solver_cpu
GPU_TARGET=solver_gpu
DIRECTIVE_TARGET=openacc_cusolver
CPU_OPTIONS=()
```

<a id="sources"></a>

## ソースと明示リンク入力

| 方式 | 教材 | benchmark |
| --- | --- | --- |
| CPU | [solver_cpu.f90](examples/solver_cpu.f90) / `solver_cpu` | [solver_cpu_bench.f90](benchmarks/solver_cpu_bench.f90) / `solver_cpu_bench` |
| CUDA | [solver_gpu.f90](examples/solver_gpu.f90) / `solver_gpu` | [solver_gpu_bench.f90](benchmarks/solver_gpu_bench.f90) / `solver_gpu_bench` |
| OpenACC | [openacc_cusolver.f90](examples/openacc_cusolver.f90) / `openacc_cusolver` | [openacc_cusolver_bench.f90](benchmarks/openacc_cusolver_bench.f90) / `openacc_cusolver_bench` |

3方式とも行列生成helper [make_dense_system.f90](../../../common/fortran/make_dense_system.f90)をリンクします。CMakeは自動的に含めます。

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

分解でA、solveでBが上書きされるため、残差確認や再試行に必要な原本を保持します。API失敗とinfoによる特異性の報告を区別してください。Aが同じでBだけ変わるなら、LUとpivotを保持してgetrsへ渡せます。毎回分解する場合とは費用が異なります。

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
  nvidia/fortran/cusolver/examples/solver_cpu.f90 common/fortran/make_dense_system.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/solver_cpu" &&
"$DIRECT_BUILD/solver_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/solver_gpu.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/solver_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/openacc_cusolver.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/openacc_cusolver" &&
"$DIRECT_BUILD/solver_gpu" &&
"$DIRECT_BUILD/openacc_cusolver"
```

<a id="examples-and-a-small-benchmark"></a>

## 教材の次に：必要なら性能測定

最小buildにはbenchmarkは含まれません。性能を測る場合は[Fortran測定手順](../README.ja.md#first-checks-and-own-measurements)でbenchmarkを有効化し、言語別configとmanifestを使います。libraryの内部CLI名は`compute`、もう一方は`end-to-end`です。教材の`main`全体とは測定範囲が異なります。

CPU教材はdgesv、benchmarkはgetrf／getrsを別々に呼びます。両scopeでrepeatは1です。

数学と計測条件は[問題仕様](../../../docs/PROJECT_SPECIFICATION.md)・[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。Fortranの結果をC/C++結果へ混ぜず、CPU組込み乱数・sumは逐次系列として扱います。

<a id="validation"></a>

## 確認済みの範囲

PegasusでNVHPCによるbuild・数値smoke・本測定の記録があります。[版と制約](../../../docs/VALIDATION_REPORT.md#fortran-production-validation)を参照してください。別のcompiler版やGPUでの動作保証ではありません。CPU要求スレッド数と実効数、リンクしたlibraryと完全な実ロード来歴は区別します。
