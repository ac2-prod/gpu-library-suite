# AMD GPUライブラリをFortranから使う

[English](README.md) | [日本語](README.ja.md)

6分野にCPU・HIP直接呼び出し・OpenMP targetの教材を用意しています。
各`examples/`に教材3本、各`benchmarks/`にbenchmark入口3本があります。
SiriusでCPU・HIP・OpenMPの36成果物のbuild、教材18本と数値smoke 72行の成功を
確認しました。rocThrust連携も含みます。続く性能pilotの108行も保存データから
独立に検証済みですが、後処理の失敗記録は保持しています。性能の本評価は未実施です。
[Siriusの記録](../../jobs/sirius/README.md#amd-fortran-first-36-artifact-build-preparation)を参照してください。
既存のAMD C/C++性能結果を、このFortran版の検証結果とは扱いません。

| 分野 | CPU教材 | HIP教材 | OpenMP教材 |
| --- | --- | --- | --- |
| [rocFFT](rocfft/examples) | `fft_cpu.f90` | `fft_gpu.f90` | `openmp_rocfft.f90` |
| [rocBLAS](rocblas/examples) | `blas_cpu.f90` | `blas_gpu.f90` | `openmp_rocblas.f90` |
| [rocSPARSE](rocsparse/examples) | `sparse_cpu.f90` | `sparse_gpu.f90` | `openmp_rocsparse.f90` |
| [rocSOLVER](rocsolver/examples) | `solver_cpu.f90` | `solver_gpu.f90` | `openmp_rocsolver.f90` |
| [rocRAND](rocrand/examples) | `rand_cpu.f90` | `rand_gpu.f90` | `openmp_rocrand.f90` |
| [rocThrust](rocthrust/examples) | `reduce_cpu.f90` | `reduce_gpu.f90` | `openmp_rocthrust.f90` |

benchmark入口は各教材名に`_bench`を加えた名前です。演算本体は同じディレクトリの
`*_workload.F90`にあります。C++ bridgeはCLI・時計・結果出力の共通処理だけを担当し、
演算をC++実装へ丸投げしていません。教材には入力初期化・エラー確認・数値検証を補い、
ライブラリ呼び出しの流れを追えるようにしています。

## CPU版から始める

リポジトリ直下から、まず外部数値ライブラリ不要の二乗和を試せます。

```bash
mkdir -p build/amd-fortran-example
gfortran -O2 amd/fortran/rocthrust/examples/reduce_cpu.f90 \
  -o build/amd-fortran-example/reduce_cpu
build/amd-fortran-example/reduce_cpu
```

期待値は`16777216`、正常終了値は0です。共有計算機ではbuild・実行をそのシステムの
計算ジョブ内で行ってください。

```bash
cmake -S . -B build/amd-fortran-cpu \
  -DGPU_SUITE_VENDOR=amd -DGPU_SUITE_SOURCE_LANGUAGE=fortran \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_HIP=OFF \
  -DGPU_SUITE_BUILD_OPENMP=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build build/amd-fortran-cpu --target reduce_cpu rand_cpu
```

CPU-onlyではROCm・hipFORT・AMDコンパイラは不要です。RAND／二乗和はFortran組込み、
FFTは単精度FFTWと`fftw3.f03`、BLAS／SolverはLP64のBLAS／LAPACK、
Sparseは明示選択したAOCL Sparseを使います。

AOCLのBLAS／LAPACKでは`GPU_SUITE_CPU_BLAS_BACKEND=GENERIC_CBLAS`、
`GPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE`と、実際のheader・library・
追加リンク入力を指定します。識別名は既存形式を使いますが、Fortran演算は
`dgemm`／`dgesv`／`dgetrf`／`dgetrs`を直接呼びます。
libFLAMEに必要なBLISもリンク順を保って明示してください。
Sparseは`GPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE`を指定します。
不足依存は対象ごとに無効化理由を表示し、自動代替はしません。

## GPU版の条件

HIP直接版はhipFORTのC互換APIを使用します。選択したFortranコンパイラと互換の
hipFORT `.mod`／`hipfort-amdgcn`が必要です。`GPU_SUITE_HIPFORT_MODULE_DIR`、
`GPU_SUITE_HIPFORT_LIBRARY`、必要なら`GPU_SUITE_ROCM_ROOT`を実環境に合わせます。
OpenMP target版にはAMD LLVM Flangと明示的な`GPU_SUITE_AMD_GPU_TARGET`も必要です。
HIPとOpenMPは別build treeに構成してください。

rocThrustは追加で`rocthrust/examples/rocthrust_wrapper.hip`をコンパイルします。
native CMake HIPには対応するClang、明示的な`CMAKE_HIP_ARCHITECTURES`、
CMake 3.29以上を指定し、`CMAKE_HIP_COMPILER`にhipccを指定しません。
手動のwrapperコンパイルではhipccを利用できます。
wrapperは3引数・整数status戻り値で、計算結果は第3引数へ書き込みます。
構成時の混合言語リンク検査も最終targetもFortranドライバでリンクし、
OpenMP版には選択したOpenMPリンクオプションを渡します。
具体的なGPU用CMake引数は[英語版のbuild例](README.md#hip-and-openmp-builds)にあります。

hipFORTの宣言が不適合または未提供の版に対応するため、`rocsolver_dgetrf`と
`rocrand_set_ordering`は呼出し元に明示した`bind(C)`宣言から同じ公開C APIを呼びます。
RANDのDEFAULT ordering指定、seed／offsetの復元、generator初期化は維持しています。

OpenMPは`map`／`use_device_addr`で明示的にデータを管理し、USMを仮定しません。
benchmarkのHIP `--device`と`OMP_DEFAULT_DEVICE`は独立しており、
mapped allocationが選択HIP deviceと一致しない場合やhost fallbackは失敗にします。
configureのリンク検査はGPU実行確認ではありません。

## 補助ソースと計測

手動buildでも以下は明示的にコンパイル・リンクします。CMakeは順序を管理します。

| 対象 | 追加ソース |
| --- | --- |
| Sparse全系列 | `rocsparse/make_poisson2d_csr.f90` |
| Sparse CPU | `rocsparse/aocl_sparse_c.f90` |
| Sparse GPU | `rocsparse/rocsparse_v2_c.f90` |
| Solver全系列 | リポジトリ直下の`common/fortran/make_dense_system.f90` |
| Thrust GPU | `rocthrust/examples/rocthrust_wrapper.hip` |
| benchmark全系列 | 各workload、共通Fortran driver・C++ bridge・共通C処理 |
| GPU benchmark | `common/hip_device_bridge.cpp` |

Sparseは0-based CSR、v2 SpMV API、csr_rowsplitです。
Solverはfactorizationのinfoを確認してからsolveへ進みます。

[smoke config](../../configs/amd/fortran/smoke.json)は6分野×3実装×2scope×
2trialの72行を要求します。各CPU backendとmanifestを一致させてください。
このconfigは実行承認や本測定計画ではありません。

表示用語は**library／end-to-end**、内部scope名は`compute`／`end-to-end`です。
[計測仕様](../../docs/BENCHMARK_PROTOCOL.md)に従い、AMD FFTはCPUを含め
各execute前の入力復元を計測外に置き、Solverは`repeat=1`とします。
外部CPUライブラリの実効スレッド数は未確認、組込みRAND／sumは逐次です。
CPUとGPUの乱数生成器は異なります。

`gpu_suite_partial_manifest`で各treeのmanifestを生成できます。
`source_language=fortran`を保持し、C/C++のmanifest・結果とは混在させません。
