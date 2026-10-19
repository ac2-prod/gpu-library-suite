# Thrust — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

transform-reduceで`sum(values[i]^2)`を計算します。変換と集約を組み合わせ、二乗値の中間配列を保存せずに求めます。

<a id="problem"></a>

## 問題と期待する結果

入力は`2^24`個のFP64の1で、二乗和の期待値は`16777216`です。CPUは逐次の組込み`sum(values*values)`です。hostへ返す結果はdouble 1個です。

Fortran教材は終了値0と`verification PASS`を確認します。固定サイズはソース内定数です。

<a id="dependencies"></a>

## 必要環境

CPUはFortranコンパイラと組込みsumだけで動きます。GPUにはThrust／CCCL・CUDA・C++ wrapper用NVCCが必要です。

CPUはGNU Fortran／NVHPC、GPUはNVHPC `nvfortran`とCUDA Toolkitを使います。Fortranの`.mod`はcompiler間で共有せず、C/C++と別treeを使います。[Fortran共通案内](../README.ja.md)にABIと数値条件をまとめています。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=thrust
CPU_TARGET=reduce_cpu
GPU_TARGET=reduce_gpu
DIRECTIVE_TARGET=openacc_thrust
CPU_OPTIONS=()
```

<a id="sources"></a>

## ソースと明示リンク入力

| 方式 | 教材 | benchmark |
| --- | --- | --- |
| CPU | [reduce_cpu.f90](examples/reduce_cpu.f90) / `reduce_cpu` | [reduce_cpu_bench.f90](benchmarks/reduce_cpu_bench.f90) / `reduce_cpu_bench` |
| CUDA | [reduce_gpu.f90](examples/reduce_gpu.f90) / `reduce_gpu` | [reduce_gpu_bench.f90](benchmarks/reduce_gpu_bench.f90) / `reduce_gpu_bench` |
| OpenACC | [openacc_thrust.f90](examples/openacc_thrust.f90) / `openacc_thrust` | [openacc_thrust_bench.f90](benchmarks/openacc_thrust_bench.f90) / `openacc_thrust_bench` |

GPUの2方式は[thrust_wrapper.cu](examples/thrust_wrapper.cu)を共用します。`bind(C)`で呼ぶ関数は`double thrust_square_sum(const double*, size_t)`です。C++例外はNaNへ変換され、Fortran側が失敗として扱います。AMDのrocThrust wrapper（3引数・int戻り値）とは別仕様です。

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

変換関数・集約演算・単位元を一組で変更します。浮動小数点の集約は順序で結果が変わり得るため、一般入力では適切な誤差許容値を使います。繰り返すGPU処理では入力を常駐させます。ディレクティブが管理するpointerをwrapperで解放せず、確保した側に所有権を残します。

Fortranの列優先配列とC ABIの型を区別します。CUDA Fortranはdevice配列、OpenACCはdata領域を使い、同じGPUライブラリへ渡します。helperやwrapperもアプリケーションのリンク入力へ含めてください。

<a id="direct-compilation"></a>

## 手動コンパイル（NVHPCでの代替手順）

通常は共通CMake手順を使います。以下はリポジトリ直下のBashで、必要な探索パスが設定済みの場合の手動手順です。CPUとGPUは別blockなので、必要な方式だけを使えます。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/thrust/examples/reduce_cpu.f90 \
  -o "$DIRECT_BUILD/reduce_cpu" &&
"$DIRECT_BUILD/reduce_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
"$CUDA_TOOLKIT_ROOT/bin/nvcc" -std=c++17 -O3 -arch="${NVCC_ARCH:?Set the nvcc architecture for your GPU}" \
  -I"${CUDA_CCCL_INCLUDE_DIR:?Set the external Toolkit directory containing thrust headers}" \
  -c nvidia/fortran/thrust/examples/thrust_wrapper.cu -o "$DIRECT_BUILD/thrust_wrapper.o" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -c++libs \
  nvidia/fortran/thrust/examples/reduce_gpu.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/reduce_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -c++libs \
  nvidia/fortran/thrust/examples/openacc_thrust.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/openacc_thrust" &&
"$DIRECT_BUILD/reduce_gpu" &&
"$DIRECT_BUILD/openacc_thrust"
```

<a id="examples-and-a-small-benchmark"></a>

## 教材の次に：必要なら性能測定

最小buildにはbenchmarkは含まれません。性能を測る場合は[Fortran測定手順](../README.ja.md#first-checks-and-own-measurements)でbenchmarkを有効化し、言語別configとmanifestを使います。libraryの内部CLI名は`compute`、もう一方は`end-to-end`です。教材の`main`全体とは測定範囲が異なります。

数学と計測条件は[問題仕様](../../../docs/PROJECT_SPECIFICATION.md)・[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。Fortranの結果をC/C++結果へ混ぜず、CPU組込み乱数・sumは逐次系列として扱います。

<a id="validation"></a>

## 確認済みの範囲

PegasusでNVHPCによるbuild・数値smoke・本測定の記録があります。[版と制約](../../../docs/VALIDATION_REPORT.md#fortran-production-validation)を参照してください。別のcompiler版やGPUでの動作保証ではありません。CPU要求スレッド数と実効数、リンクしたlibraryと完全な実ロード来歴は区別します。
