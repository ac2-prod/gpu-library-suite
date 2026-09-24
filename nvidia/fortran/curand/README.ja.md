# cuRAND — NVIDIA Fortran

[English](README.md) | [日本語](README.ja.md)

[共通のFortran利用案内](../README.ja.md)に必要環境、分離CMake build、flags、完了したPegasus実機検証の範囲と留保をまとめています。

| 方式 | 教材 | Benchmark |
| --- | --- | --- |
| CPU | [rand_cpu.f90](examples/rand_cpu.f90) | [rand_cpu_bench.f90](benchmarks/rand_cpu_bench.f90) |
| CUDA | [rand_gpu.f90](examples/rand_gpu.f90) | [rand_gpu_bench.f90](benchmarks/rand_gpu_bench.f90) |
| OpenACC | [openacc_curand.f90](examples/openacc_curand.f90) | [openacc_curand_bench.f90](benchmarks/openacc_curand_bench.f90) |

FP64一様乱数の生成です。CPUはrandom_seed(size=...)で得た長さのseed配列を1234で埋め、random_numberを使います。C++ MTへ置換しません。アルゴリズムと数列はFortranコンパイラ依存で、CURAND_RNG_PSEUDO_DEFAULTとは分布生成の仕事を比較します。区間はCPUが[0,1)、GPUが(0,1]です。finite・範囲・平均・0.5周りの2次中心モーメントを設定済みsigma閾値で検証します。computeではrepeat中に数列を進め、各trial前にseed/offsetを復元し、E2Eは各pipelineで設定します。CPU benchmarkのseedはINT_MAX以下で、offsetは一定サイズずつ乱数を消費して進めます。

## 直接コンパイル

推奨は[共通CMake手順](../README.ja.md#separate-gpu-build-trees)です。以下は同じToolkit環境でリポジトリrootから実行するNVHPC/Linux用コマンドで、実機ビルド済みの主張ではありません。providerのinclude/library directoryは実際の場所を指定します。helper・wrapperは明示的なリンク入力です。依存パッケージの導入は行いません。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-fortran-example.XXXXXX")"
F90FLAGS=(-O3 -Kieee -Mnoflushz -Mnodaz)
GPUFLAGS=(-cuda "-gpu=${NVHPC_GPU_TARGET:?Set the actual target},mem:separate")
unset NVCOMPILER_FPU_STATE
nvfortran "${F90FLAGS[@]}" nvidia/fortran/curand/examples/rand_cpu.f90 \
  -o "$DIRECT_BUILD/rand_cpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -cudalib=curand \
  nvidia/fortran/curand/examples/rand_gpu.f90 -o "$DIRECT_BUILD/rand_gpu"
nvfortran "${F90FLAGS[@]}" "${GPUFLAGS[@]}" -acc=gpu -cudalib=curand \
  nvidia/fortran/curand/examples/openacc_curand.f90 -o "$DIRECT_BUILD/openacc_curand"
```

## 教材と小規模benchmark

CMake手順で設定した2つのbuild変数を使います。各教材の終了0とverification PASSを確認します。直接コンパイルした場合はDIRECT_BUILD内の同名実行ファイルを使います。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/curand/rand_cpu"
"$CPU_CUDA_BUILD/nvidia/fortran/curand/rand_gpu"
"$OPENACC_BUILD/nvidia/fortran/curand/openacc_curand"
```

RUN_DIRは新規出力先です。次は単独CPU診断で、campaign入力ではありません。

```bash
"$CPU_CUDA_BUILD/nvidia/fortran/curand/rand_cpu_bench" \
  --size 65536 --seed 1234 --offset 0 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --cpu-threads 1 --cpu-backend cpu-fortran-random-serial \
  --cpu-parallelism serial --cpu-threads-effective 1 \
  --output "$RUN_DIR/standalone-curand-cpu-compute.jsonl" --format jsonl
```

E2Eは--scopeをend-to-endへ変え、新しい出力名を使います。GPU診断では表の対応するbenchmarkパス（CUDAはCPU_CUDA_BUILD、OpenACCはOPENACC_BUILD）に切り替え、CPU専用引数を外し、同じ問題・verification・反復条件を維持します。E2E診断はrepeat=1から始めます。

[自己測定→検証→集計→図生成](../README.ja.md#first-checks-and-own-measurements)は既存runnerを使用します。compute/E2Eの範囲、入力復元、許容誤差の正本は[測定規約](../../../docs/BENCHMARK_PROTOCOL.ja.md)です。Fortran言語識別子を保持し、C/C++測定値と混ぜません。
