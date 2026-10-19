# cuRAND — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

FP64の一様乱数を生成します。シミュレーション入力や確率的アルゴリズムで使うgeneratorと出力bufferの扱いを学べます。

<a id="problem"></a>

## 問題と期待する結果

seed 1234で`2^24`個を生成します。CPUはFortran組込み乱数、GPUはライブラリ既定の疑似乱数generatorです。CPUの区間は[0,1)、GPUは(0,1]です。同じ分布の比較であり、同じ乱数列ではありません。

Fortran教材は終了値0と`verification PASS`を確認します。固定サイズはソース内定数です。

<a id="dependencies"></a>

## 必要環境

CPUはFortranコンパイラと組込み乱数だけで動きます。GPUにはcuRANDが必要です。

CPUはGNU Fortran／NVHPC、GPUはNVHPC `nvfortran`とCUDA Toolkitを使います。Fortranの`.mod`はcompiler間で共有せず、C/C++と別treeを使います。[Fortran共通案内](../README.ja.md)にABIと数値条件をまとめています。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=curand
CPU_TARGET=rand_cpu
GPU_TARGET=rand_gpu
DIRECTIVE_TARGET=openacc_curand
CPU_OPTIONS=()
```

<a id="sources"></a>

## ソースと明示リンク入力

| 方式 | 教材 | benchmark |
| --- | --- | --- |
| CPU | [rand_cpu.f90](examples/rand_cpu.f90) / `rand_cpu` | [rand_cpu_bench.f90](benchmarks/rand_cpu_bench.f90) / `rand_cpu_bench` |
| CUDA | [rand_gpu.f90](examples/rand_gpu.f90) / `rand_gpu` | [rand_gpu_bench.f90](benchmarks/rand_gpu_bench.f90) / `rand_gpu_bench` |
| OpenACC | [openacc_curand.f90](examples/openacc_curand.f90) / `openacc_curand` | [openacc_curand_bench.f90](benchmarks/openacc_curand_bench.f90) / `openacc_curand_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

用途に合うgeneratorとseed／streamの方針を選びます。同じ列の再生成を意図しない限り、毎回seedを戻しません。継続する列ではgeneratorとbufferを再利用します。CPUとGPUのアルゴリズムは異なるため、要素一致ではなく範囲・分布統計で確認します。CPU教材は逐次です。

Fortranの列優先配列とC ABIの型を区別します。CUDA Fortranはdevice配列、OpenACCはdata領域を使い、同じGPUライブラリへ渡します。helperやwrapperもアプリケーションのリンク入力へ含めてください。

<a id="direct-compilation"></a>

## 手動コンパイル（NVHPCでの代替手順）

通常は共通CMake手順を使います。以下はリポジトリ直下のBashで、必要な探索パスが設定済みの場合の手動手順です。CPUとGPUは別blockなので、必要な方式だけを使えます。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/curand/examples/rand_cpu.f90 \
  -o "$DIRECT_BUILD/rand_cpu" &&
"$DIRECT_BUILD/rand_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=curand \
  nvidia/fortran/curand/examples/rand_gpu.f90 -o "$DIRECT_BUILD/rand_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=curand \
  nvidia/fortran/curand/examples/openacc_curand.f90 -o "$DIRECT_BUILD/openacc_curand" &&
"$DIRECT_BUILD/rand_gpu" &&
"$DIRECT_BUILD/openacc_curand"
```

<a id="examples-and-a-small-benchmark"></a>

## 教材の次に：必要なら性能測定

最小buildにはbenchmarkは含まれません。性能を測る場合は[Fortran測定手順](../README.ja.md#first-checks-and-own-measurements)でbenchmarkを有効化し、言語別configとmanifestを使います。libraryの内部CLI名は`compute`、もう一方は`end-to-end`です。教材の`main`全体とは測定範囲が異なります。

数学と計測条件は[問題仕様](../../../docs/PROJECT_SPECIFICATION.md)・[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。Fortranの結果をC/C++結果へ混ぜず、CPU組込み乱数・sumは逐次系列として扱います。

<a id="validation"></a>

## 確認済みの範囲

PegasusでNVHPCによるbuild・数値smoke・本測定の記録があります。[版と制約](../../../docs/VALIDATION_REPORT.md#fortran-production-validation)を参照してください。別のcompiler版やGPUでの動作保証ではありません。CPU要求スレッド数と実効数、リンクしたlibraryと完全な実ロード来歴は区別します。
