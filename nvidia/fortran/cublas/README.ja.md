# cuBLAS — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

[共通のFortran利用案内](../README.ja.md)に必要環境、分離CMake build、flags、完了したPegasus実機検証の範囲と留保をまとめています。

| 方式 | 教材 | Benchmark |
| --- | --- | --- |
| CPU | [blas_cpu.f90](examples/blas_cpu.f90) | [blas_cpu_bench.f90](benchmarks/blas_cpu_bench.f90) |
| CUDA | [blas_gpu.f90](examples/blas_gpu.f90) | [blas_gpu_bench.f90](benchmarks/blas_gpu_bench.f90) |
| OpenACC | [openacc_cublas.f90](examples/openacc_cublas.f90) | [openacc_cublas_bench.f90](benchmarks/openacc_cublas_bench.f90) |

FP64列優先のC=alpha*A*B+beta*Cで、A/B/Cの初期値はすべて1です。CPUはoneMKLのFortran dgemm、GPUはcublas_v2を呼びます。Cは入力でもあるためOpenACCはcopyを使い、copyoutだけにしません。compute反復では漸化式どおり蓄積し、E2Eは各区間外で元のCへ戻します。検証も反復に対応します。benchmarkの--m/--n/--kでは非正方問題も指定できます。

## 直接コンパイル

推奨は[共通CMake手順](../README.ja.md#separate-gpu-build-trees)です。以下は同じToolkit環境でリポジトリrootから実行するNVHPC/Linux用コマンドで、実機ビルド済みの主張ではありません。providerのinclude/library directoryは実際の場所を指定します。helper・wrapperは明示的なリンク入力です。依存パッケージの導入は行いません。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cublas/examples/blas_cpu.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/blas_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cublas \
  nvidia/fortran/cublas/examples/blas_gpu.f90 -o "$DIRECT_BUILD/blas_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cublas \
  nvidia/fortran/cublas/examples/openacc_cublas.f90 -o "$DIRECT_BUILD/openacc_cublas"
```

## 教材と小規模benchmark

CMake手順で設定した2つのbuild変数を使います。各教材の終了0とverification PASSを確認します。直接コンパイルした場合はDIRECT_BUILD内の同名実行ファイルを使います。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cublas/blas_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cublas/blas_gpu"
"$OPENACC_BUILD/nvidia/fortran/cublas/openacc_cublas"
```

RUN_DIRは新規出力先です。次は単独CPU診断で、campaign入力ではありません。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cublas/blas_cpu_bench" \
  --size 32 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-onemkl \
  --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cublas-cpu-compute.jsonl" --format jsonl
```

E2Eは--scopeをend-to-endへ変え、新しい出力名を使います。GPU診断では表の対応するbenchmarkパス（CUDAはCPU_CUDA_BUILD、OpenACCはOPENACC_BUILD）に切り替え、CPU専用引数を外し、同じ問題・verification・反復条件を維持します。E2E診断はrepeat=1から始めます。

[自己測定→検証→集計→図生成](../README.ja.md#first-checks-and-own-measurements)は既存runnerを使用します。compute/E2Eの範囲、入力復元、許容誤差の正本は[測定規約](../../../docs/BENCHMARK_PROTOCOL.ja.md)です。Fortran言語識別子を保持し、C/C++測定値と混ぜません。
