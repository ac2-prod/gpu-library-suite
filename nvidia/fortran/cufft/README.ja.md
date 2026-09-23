# cuFFT — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

[共通のFortran利用案内](../README.ja.md)に必要環境、分離CMake build、flags、検証状態をまとめています。GPU実行は未確認です。

| 方式 | 教材 | Benchmark |
| --- | --- | --- |
| CPU | [fft_cpu.f90](examples/fft_cpu.f90) | [fft_cpu_bench.f90](benchmarks/fft_cpu_bench.f90) |
| CUDA | [fft_gpu.f90](examples/fft_gpu.f90) | [fft_gpu_bench.f90](benchmarks/fft_gpu_bench.f90) |
| OpenACC | [openacc_cufft.f90](examples/openacc_cufft.f90) | [openacc_cufft_bench.f90](benchmarks/openacc_cufft_bench.f90) |

FP32複素数のout-of-place batched 1D C2Cです。入力は全要素1で、DCがFFT長、それ以外が許容誤差内の0か確認します。CPU教材は逐次のFFTW_ESTIMATEです。benchmarkはthreads APIが利用可能ならcpu-fftw-threadedも選べ、plan作成前に設定します。plan・転送はcomputeの外、E2Eの各区間内です。p.13の未完のstream補足は対象外です。

## 直接コンパイル

推奨は[共通CMake手順](../README.ja.md#separate-gpu-build-trees)です。以下は同じToolkit環境でリポジトリrootから実行するNVHPC/Linux用コマンドで、実機ビルド済みの主張ではありません。providerのinclude/library directoryは実際の場所を指定します。helper・wrapperは明示的なリンク入力です。依存パッケージの導入は行いません。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${FFTW_INCLUDE_DIR:?Set the directory containing fftw3.f03}" \
  nvidia/fortran/cufft/examples/fft_cpu.f90 \
  -L"${FFTW_LIBDIR:?Set the directory containing libfftw3f}" -lfftw3f \
  -o "$DIRECT_BUILD/fft_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cufft \
  nvidia/fortran/cufft/examples/fft_gpu.f90 -o "$DIRECT_BUILD/fft_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cufft \
  nvidia/fortran/cufft/examples/openacc_cufft.f90 -o "$DIRECT_BUILD/openacc_cufft"
```

## 教材と小規模benchmark

CMake手順で設定した2つのbuild変数を使います。各教材の終了0とverification PASSを確認します。直接コンパイルした場合はDIRECT_BUILD内の同名実行ファイルを使います。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cufft/fft_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cufft/fft_gpu"
"$OPENACC_BUILD/nvidia/fortran/cufft/openacc_cufft"
```

RUN_DIRは新規出力先です。次は単独CPU診断で、campaign入力ではありません。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cufft/fft_cpu_bench" \
  --size 64 --batch 4 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-fftw-serial \
  --cpu-parallelism serial --cpu-threads-effective 1 \
  --output "$RUN_DIR/standalone-cufft-cpu-compute.jsonl" --format jsonl
```

E2Eは--scopeをend-to-endへ変え、新しい出力名を使います。GPU診断では表の対応するbenchmarkパス（CUDAはCPU_CUDA_BUILD、OpenACCはOPENACC_BUILD）に切り替え、CPU専用引数を外し、同じ問題・verification・反復条件を維持します。E2E診断はrepeat=1から始めます。

[自己測定→検証→集計→図生成](../README.ja.md#first-checks-and-own-measurements)は既存runnerを使用します。compute/E2Eの範囲、入力復元、許容誤差の正本は[測定規約](../../../docs/BENCHMARK_PROTOCOL.ja.md)です。Fortran言語識別子を保持し、C/C++測定値と混ぜません。
