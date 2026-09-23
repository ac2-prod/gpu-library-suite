# cuSOLVER — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

[共通のFortran利用案内](../README.ja.md)に必要環境、分離CMake build、flags、検証状態をまとめています。GPU実行は未確認です。

| 方式 | 教材 | Benchmark |
| --- | --- | --- |
| CPU | [solver_cpu.f90](examples/solver_cpu.f90) | [solver_cpu_bench.f90](benchmarks/solver_cpu_bench.f90) |
| CUDA | [solver_gpu.f90](examples/solver_gpu.f90) | [solver_gpu_bench.f90](benchmarks/solver_gpu_bench.f90) |
| OpenACC | [openacc_cusolver.f90](examples/openacc_cusolver.f90) | [openacc_cusolver_bench.f90](benchmarks/openacc_cusolver_bench.f90) |

FP64のAX=Bで、A=N*I+ones、Bの全要素が2*N、正解は1です。外部helperで教材の省略部分を同じ問題のまま補完します。CPU教材はdgesv、benchmarkはGPUの段階に合わせdgetrf→dgetrsです。GPUのgetrf/getrs infoは分離して保持します。教材ではfactorization infoをsolve前に確認し、benchmarkでは計時内の段階ごとのbarrierを増やさず、同期・転送後に両infoを確認します。解の誤差と元の系の残差を検証します。両scopeでrepeatは1に限定します。

## 直接コンパイル

推奨は[共通CMake手順](../README.ja.md#separate-gpu-build-trees)です。以下は同じToolkit環境でリポジトリrootから実行するNVHPC/Linux用コマンドで、実機ビルド済みの主張ではありません。providerのinclude/library directoryは実際の場所を指定します。helper・wrapperは明示的なリンク入力です。依存パッケージの導入は行いません。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cusolver/examples/solver_cpu.f90 common/fortran/make_dense_system.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/solver_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/solver_gpu.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/solver_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusolver \
  nvidia/fortran/cusolver/examples/openacc_cusolver.f90 common/fortran/make_dense_system.f90 -o "$DIRECT_BUILD/openacc_cusolver"
```

## 教材と小規模benchmark

CMake手順で設定した2つのbuild変数を使います。各教材の終了0とverification PASSを確認します。直接コンパイルした場合はDIRECT_BUILD内の同名実行ファイルを使います。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusolver/solver_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cusolver/solver_gpu"
"$OPENACC_BUILD/nvidia/fortran/cusolver/openacc_cusolver"
```

RUN_DIRは新規出力先です。次は単独CPU診断で、campaign入力ではありません。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusolver/solver_cpu_bench" \
  --size 32 --nrhs 2 --warmup 1 --repeat 1 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-onemkl \
  --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cusolver-cpu-compute.jsonl" --format jsonl
```

E2Eは--scopeをend-to-endへ変え、新しい出力名を使います。GPU診断では表の対応するbenchmarkパス（CUDAはCPU_CUDA_BUILD、OpenACCはOPENACC_BUILD）に切り替え、CPU専用引数を外し、同じ問題・verification・反復条件を維持します。repeatは常に1です。

[自己測定→検証→集計→図生成](../README.ja.md#first-checks-and-own-measurements)は既存runnerを使用します。compute/E2Eの範囲、入力復元、許容誤差の正本は[測定規約](../../../docs/BENCHMARK_PROTOCOL.ja.md)です。Fortran言語識別子を保持し、C/C++測定値と混ぜません。
