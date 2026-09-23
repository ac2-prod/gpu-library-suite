# cuSPARSE — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

[共通のFortran利用案内](../README.ja.md)に必要環境、分離CMake build、flags、検証状態をまとめています。GPU実行は未確認です。

| 方式 | 教材 | Benchmark |
| --- | --- | --- |
| CPU | [sparse_cpu.f90](examples/sparse_cpu.f90) | [sparse_cpu_bench.f90](benchmarks/sparse_cpu_bench.f90) |
| CUDA | [sparse_gpu.f90](examples/sparse_gpu.f90) | [sparse_gpu_bench.f90](benchmarks/sparse_gpu_bench.f90) |
| OpenACC | [openacc_cusparse.f90](examples/openacc_cusparse.f90) | [openacc_cusparse_bench.f90](benchmarks/openacc_cusparse_bench.f90) |

FP64の2D 5点Poisson SpMVです。外部helperは既存の対角4・隣接-1、昇順の1始まりCSR、N=nx*ny、nnz=5*nx*ny-2*nx-2*nyを実装します。境界を越える辺は作りません。CPUはoneMKL Sparse BLAS、GPUはINDEX_BASE_ONEのgeneric cuSPARSE descriptorです。初期x=y=1を保持し、OpenACCのyはcopyします。workspaceは明示的device allocationです。benchmarkのdescriptor解析・workspace準備はE2Eだけで計時します。

## 直接コンパイル

推奨は[共通CMake手順](../README.ja.md#separate-gpu-build-trees)です。以下は同じToolkit環境でリポジトリrootから実行するNVHPC/Linux用コマンドで、実機ビルド済みの主張ではありません。providerのinclude/library directoryは実際の場所を指定します。helper・wrapperは明示的なリンク入力です。依存パッケージの導入は行いません。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" -module "$DIRECT_BUILD" \
  -I"${MKLROOT:?Set the installed oneMKL root}/include" \
  nvidia/fortran/cusparse/examples/sparse_cpu.f90 common/fortran/make_poisson2d_csr.f90 \
  -L"${MKL_LIBDIR:?Set the installed oneMKL library directory}" -lmkl_rt -lpthread -lm -ldl \
  -o "$DIRECT_BUILD/sparse_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/sparse_gpu.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/sparse_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=cusparse \
  nvidia/fortran/cusparse/examples/openacc_cusparse.f90 common/fortran/make_poisson2d_csr.f90 -o "$DIRECT_BUILD/openacc_cusparse"
```

## 教材と小規模benchmark

CMake手順で設定した2つのbuild変数を使います。各教材の終了0とverification PASSを確認します。直接コンパイルした場合はDIRECT_BUILD内の同名実行ファイルを使います。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusparse/sparse_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/cusparse/sparse_gpu"
"$OPENACC_BUILD/nvidia/fortran/cusparse/openacc_cusparse"
```

RUN_DIRは新規出力先です。次は単独CPU診断で、campaign入力ではありません。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/cusparse/sparse_cpu_bench" \
  --nx 8 --ny 8 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-onemkl \
  --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cusparse-cpu-compute.jsonl" --format jsonl
```

E2Eは--scopeをend-to-endへ変え、新しい出力名を使います。GPU診断では表の対応するbenchmarkパス（CUDAはCPU_CUDA_BUILD、OpenACCはOPENACC_BUILD）に切り替え、CPU専用引数を外し、同じ問題・verification・反復条件を維持します。E2E診断はrepeat=1から始めます。

[自己測定→検証→集計→図生成](../README.ja.md#first-checks-and-own-measurements)は既存runnerを使用します。compute/E2Eの範囲、入力復元、許容誤差の正本は[測定規約](../../../docs/BENCHMARK_PROTOCOL.ja.md)です。Fortran言語識別子を保持し、C/C++測定値と混ぜません。
