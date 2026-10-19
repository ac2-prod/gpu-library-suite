# cuFFT — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

長さ1,024のFP32複素数FFTを4,096本まとめてforward変換します。多数の信号をまとめて処理するアプリケーションで、batched FFTの呼び出し方を学べます。

<a id="problem"></a>

## 問題と期待する結果

out-of-placeの1D C2Cです。複素数をinterleaved形式で連続配置し、入力をすべて`1+0i`にします。各FFTのDC成分は`1024+0i`、他は0で、ソース内の許容誤差で確認します。

Fortran教材は終了値0と`verification PASS`を確認します。固定サイズはソース内定数です。

<a id="dependencies"></a>

## 必要環境

CPUには単精度FFTWの`fftw3.h`・`libfftw3f`・`fftw3.f03`が必要です。通常の探索範囲外なら`FFTW_ROOT`でprefixを指定します。スレッド版はbenchmark用の任意依存で、逐次教材には不要です。

CPUはGNU Fortran／NVHPC、GPUはNVHPC `nvfortran`とCUDA Toolkitを使います。Fortranの`.mod`はcompiler間で共有せず、C/C++と別treeを使います。[Fortran共通案内](../README.ja.md)にABIと数値条件をまとめています。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=nvidia
SOURCE_LANGUAGE=fortran
LIBRARY=cufft
CPU_TARGET=fft_cpu
GPU_TARGET=fft_gpu
DIRECTIVE_TARGET=openacc_cufft
CPU_OPTIONS=()
```

<a id="sources"></a>

## ソースと明示リンク入力

| 方式 | 教材 | benchmark |
| --- | --- | --- |
| CPU | [fft_cpu.f90](examples/fft_cpu.f90) / `fft_cpu` | [fft_cpu_bench.f90](benchmarks/fft_cpu_bench.f90) / `fft_cpu_bench` |
| CUDA | [fft_gpu.f90](examples/fft_gpu.f90) / `fft_gpu` | [fft_gpu_bench.f90](benchmarks/fft_gpu_bench.f90) / `fft_gpu_bench` |
| OpenACC | [openacc_cufft.f90](examples/openacc_cufft.f90) / `openacc_cufft` | [openacc_cufft_bench.f90](benchmarks/openacc_cufft_bench.f90) / `openacc_cufft_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

精度・変換方向・stride・distance・batchを実際のbufferに合わせます。planは形状と配置が合う間だけ再利用します。FFTW教材は逐次の`FFTW_ESTIMATE`で、スレッド版benchmarkとは別です。

Fortranの列優先配列とC ABIの型を区別します。CUDA Fortranはdevice配列、OpenACCはdata領域を使い、同じGPUライブラリへ渡します。helperやwrapperもアプリケーションのリンク入力へ含めてください。

<a id="direct-compilation"></a>

## 手動コンパイル（NVHPCでの代替手順）

通常は共通CMake手順を使います。以下はリポジトリ直下のBashで、必要な探索パスが設定済みの場合の手動手順です。CPUとGPUは別blockなので、必要な方式だけを使えます。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${FFTW_INCLUDE_DIR:?Set the directory containing fftw3.f03}" \
  nvidia/fortran/cufft/examples/fft_cpu.f90 \
  -L"${FFTW_LIBDIR:?Set the directory containing libfftw3f}" -lfftw3f \
  -o "$DIRECT_BUILD/fft_cpu" &&
"$DIRECT_BUILD/fft_cpu"
```

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-gpu.XXXXXX")" || exit 1
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
: "${CUDA_TOOLKIT_ROOT:?Set your CUDA Toolkit prefix}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set your NVHPC GPU target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cufft \
  nvidia/fortran/cufft/examples/fft_gpu.f90 -o "$DIRECT_BUILD/fft_gpu" &&
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cufft \
  nvidia/fortran/cufft/examples/openacc_cufft.f90 -o "$DIRECT_BUILD/openacc_cufft" &&
"$DIRECT_BUILD/fft_gpu" &&
"$DIRECT_BUILD/openacc_cufft"
```

<a id="examples-and-a-small-benchmark"></a>

## 教材の次に：必要なら性能測定

最小buildにはbenchmarkは含まれません。性能を測る場合は[Fortran測定手順](../README.ja.md#first-checks-and-own-measurements)でbenchmarkを有効化し、言語別configとmanifestを使います。libraryの内部CLI名は`compute`、もう一方は`end-to-end`です。教材の`main`全体とは測定範囲が異なります。

数学と計測条件は[問題仕様](../../../docs/PROJECT_SPECIFICATION.md)・[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。Fortranの結果をC/C++結果へ混ぜず、CPU組込み乱数・sumは逐次系列として扱います。

<a id="validation"></a>

## 確認済みの範囲

PegasusでNVHPCによるbuild・数値smoke・本測定の記録があります。[版と制約](../../../docs/VALIDATION_REPORT.md#fortran-production-validation)を参照してください。別のcompiler版やGPUでの動作保証ではありません。CPU要求スレッド数と実効数、リンクしたlibraryと完全な実ロード来歴は区別します。
