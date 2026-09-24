# Thrust — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

[共通のFortran利用案内](../README.ja.md)に必要環境、分離CMake build、flags、完了したPegasus実機検証の範囲と留保をまとめています。

| 方式 | 教材 | Benchmark |
| --- | --- | --- |
| CPU | [reduce_cpu.f90](examples/reduce_cpu.f90) | [reduce_cpu_bench.f90](benchmarks/reduce_cpu_bench.f90) |
| CUDA | [reduce_gpu.f90](examples/reduce_gpu.f90) | [reduce_gpu_bench.f90](benchmarks/reduce_gpu_bench.f90) |
| OpenACC | [openacc_thrust.f90](examples/openacc_thrust.f90) | [openacc_thrust_bench.f90](benchmarks/openacc_thrust_bench.f90) |

全要素1のFP64配列の平方和です。CPUはFortran sum(values*values)を保持し、OpenMPループ・STLへの置換・自動GPU offloadを入れません。両GPU版は提供されたextern-C thrust_square_sumをbind(C)から呼び、transform_reduceを含むwrapperを共用します。C++例外はstderrへ診断しNaNを返してFortran側で拒否します。benchmarkの同期は計時境界です。教材の同期方式は保持し、ホストスカラーが返ることを理由にbenchmarkの完了確認を省きません。

## 直接コンパイル

推奨は[共通CMake手順](../README.ja.md#separate-gpu-build-trees)です。以下は同じToolkit環境でリポジトリrootから実行するNVHPC/Linux用コマンドで、実機ビルド済みの主張ではありません。providerのinclude/library directoryは実際の場所を指定します。helper・wrapperは明示的なリンク入力です。依存パッケージの導入は行いません。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/thrust/examples/reduce_cpu.f90 \
  -o "$DIRECT_BUILD/reduce_cpu"
nvcc -std=c++17 -O3 -arch="${NVCC_ARCH:?Set the NVCC sm target}" \
  -I"${CUDA_CCCL_INCLUDE_DIR:?Set the external Toolkit directory containing thrust headers}" \
  -c nvidia/fortran/thrust/examples/thrust_wrapper.cu -o "$DIRECT_BUILD/thrust_wrapper.o"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -c++libs \
  nvidia/fortran/thrust/examples/reduce_gpu.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/reduce_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -c++libs \
  nvidia/fortran/thrust/examples/openacc_thrust.f90 "$DIRECT_BUILD/thrust_wrapper.o" -o "$DIRECT_BUILD/openacc_thrust"
```

## 教材と小規模benchmark

CMake手順で設定した2つのbuild変数を使います。各教材の終了0とverification PASSを確認します。直接コンパイルした場合はDIRECT_BUILD内の同名実行ファイルを使います。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/thrust/reduce_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/thrust/reduce_gpu"
"$OPENACC_BUILD/nvidia/fortran/thrust/openacc_thrust"
```

RUN_DIRは新規出力先です。次は単独CPU診断で、campaign入力ではありません。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/thrust/reduce_cpu_bench" \
  --size 4096 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-fortran-sum-serial \
  --cpu-parallelism serial --cpu-threads-effective 1 \
  --output "$RUN_DIR/standalone-thrust-cpu-compute.jsonl" --format jsonl
```

E2Eは--scopeをend-to-endへ変え、新しい出力名を使います。GPU診断では表の対応するbenchmarkパス（CUDAはCPU_CUDA_BUILD、OpenACCはOPENACC_BUILD）に切り替え、CPU専用引数を外し、同じ問題・verification・反復条件を維持します。E2E診断はrepeat=1から始めます。

[自己測定→検証→集計→図生成](../README.ja.md#first-checks-and-own-measurements)は既存runnerを使用します。compute/E2Eの範囲、入力復元、許容誤差の正本は[測定規約](../../../docs/BENCHMARK_PROTOCOL.ja.md)です。Fortran言語識別子を保持し、C/C++測定値と混ぜません。
