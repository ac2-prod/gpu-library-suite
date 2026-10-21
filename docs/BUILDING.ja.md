# 教材サンプルのビルドと実行

[English](BUILDING.md) | [日本語](BUILDING.ja.md)

1つの教材を動かすための手順です。ベンチマーク・全件テスト・測定campaignは構築しません。既に必要なコンパイラ・ライブラリがあるLinux環境を想定します。依存の自動インストールは行いません。共有計算機では計算ジョブ内で実行し、[Pegasus](PEGASUS_EXECUTION.md)／[Sirius](../jobs/sirius/README.md)など、そのシステムの規則に従ってください。

<a id="select"></a>

## 1. サンプルと環境を選ぶ

[一覧](../README.ja.md#start-here)から各ライブラリのREADMEを開き、「最小ビルドの選択」の変数ブロックをBashへ設定してください。`CPU_OPTIONS`はそのCPUライブラリを明示選択する配列です。以下はすべてリポジトリ直下・同じBashセッションで実行します。必要な方式の節だけを使い、失敗したコマンドの後は停止してください。

**AMD Fortran**には専用の[CPU・GPUビルド手順](../amd/fortran/README.ja.md)を使います。
互換のhipFORT module、Fortranリンカー、rocThrustの補助コードを選択するためです。
本書の5・6節に示すAMD GPUコマンドはC/C++用です。

| 方式 | 必要環境 |
| --- | --- |
| C/C++ CPU | CMake 3.20+、C17／C++17、演算ごとのCPU依存 |
| Fortran CPU | 上記＋GNU FortranまたはNVHPC。`F90_COMPILER=gfortran`または`nvfortran` |
| NVIDIA C/C++ CUDA | NVIDIA GPU／ドライバ、CUDA Toolkit／nvcc、対応するhost compiler |
| NVIDIA C/C++ OpenACC | 同じCUDA Toolkit＋NVHPC nvc／nvc++ |
| NVIDIA Fortran GPU | NVHPC nvfortran、CUDA Toolkit。ThrustにはNVCC／CCCLも必要 |
| AMD C/C++ HIP | AMD GPU／ドライバ、ROCm、HIP対応Clang、CMake 3.21+ |
| AMD C/C++ OpenMP target | ROCmと対応するAMD Clang。rocThrustにはHIP ClangとCMake 3.21+も必要 |
| AMD Fortran GPU | 互換のFortranコンパイラとhipFORT。OpenMP targetはAMD LLVM Flang、rocThrustはnative HIP ClangとCMake 3.29+も必要 |

compiler・GPU target・ROCm／CUDAのprefixは自分の環境で確認した値を使います。別のvendor・言語・compiler・実行方式への変更には新しいbuild treeを作ります。初期configure後にcompilerを切り替えないでください。
```bash
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-examples.XXXXXX")" || exit 1
COMMON=(-DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_EXAMPLES=ON
  -DGPU_SUITE_BUILD_BENCHMARKS=OFF -DGPU_SUITE_BUILD_TESTS=OFF
  "-DGPU_SUITE_VENDOR=$VENDOR" "-DGPU_SUITE_SOURCE_LANGUAGE=$SOURCE_LANGUAGE")
if [ "$VENDOR" = amd ]; then
  COMMON+=("-DGPU_SUITE_AMD_LIBRARIES=$LIBRARY")
fi
```


<a id="cpu"></a>

## 2. CPUだけで実行

GPUを有効化せず、選択したCPU教材だけをbuild・実行します。
```bash
F90_OPTIONS=()
if [ "$SOURCE_LANGUAGE" = fortran ]; then
  F90_OPTIONS=("-DCMAKE_Fortran_COMPILER=${F90_COMPILER:?Set gfortran or nvfortran}")
  unset NVCOMPILER_FPU_STATE
fi
cmake -S . -B "$BUILD_ROOT/cpu" "${COMMON[@]}" "${F90_OPTIONS[@]}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_HIP=OFF \
  -DGPU_SUITE_BUILD_OPENMP=OFF "${CPU_OPTIONS[@]}" &&
cmake --build "$BUILD_ROOT/cpu" --target "$CPU_TARGET" --parallel 2 &&
"$BUILD_ROOT/cpu/$VENDOR/$SOURCE_LANGUAGE/$LIBRARY/$CPU_TARGET"
```


<a id="cuda"></a>

## 3. NVIDIA CUDA／CUDA Fortran

CPU数値ライブラリは不要です。FortranではNVHPCとNVCCが同じ外部Toolkitを使うようにします。GCC／G++はそのToolkitの対応版を選んでください。
```bash
test "$VENDOR" = nvidia || exit 1
: "${CUDA_TOOLKIT_ROOT:?Set the installed CUDA Toolkit prefix}"
: "${CUDA_ARCHITECTURES:?Set the CMake architecture for your GPU}"
CUDA_F90_OPTIONS=()
if [ "$SOURCE_LANGUAGE" = fortran ]; then
  : "${NVHPC_GPU_TARGET:?Set the NVHPC GPU target}"
  export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
  unset NVCOMPILER_FPU_STATE
  CUDA_F90_OPTIONS=(-DCMAKE_Fortran_COMPILER=nvfortran
    "-DGPU_SUITE_NVHPC_GPU_TARGET=$NVHPC_GPU_TARGET")
fi
cmake -S . -B "$BUILD_ROOT/cuda" "${COMMON[@]}" "${CUDA_F90_OPTIONS[@]}" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF &&
cmake --build "$BUILD_ROOT/cuda" --target "$GPU_TARGET" --parallel 2 &&
"$BUILD_ROOT/cuda/nvidia/$SOURCE_LANGUAGE/$LIBRARY/$GPU_TARGET"
```


<a id="openacc"></a>

## 4. NVIDIA OpenACC

NVHPCとCMakeのCUDA Toolkitを一致させます。ThrustのFortran wrapperもCMakeが明示的にcompile・linkします。
```bash
test "$VENDOR" = nvidia || exit 1
: "${CUDA_TOOLKIT_ROOT:?Set the installed CUDA Toolkit prefix}"
: "${NVHPC_GPU_TARGET:?Set the NVHPC GPU target}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
ACC_COMPILERS=(-DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++)
if [ "$SOURCE_LANGUAGE" = fortran ]; then
  : "${CUDA_ARCHITECTURES:?Set the CMake CUDA architecture for the Thrust wrapper}"
  unset NVCOMPILER_FPU_STATE
  ACC_COMPILERS=(-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
    -DCMAKE_Fortran_COMPILER=nvfortran
    "-DCMAKE_CUDA_COMPILER=$CUDA_TOOLKIT_ROOT/bin/nvcc"
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES")
fi
cmake -S . -B "$BUILD_ROOT/openacc" "${COMMON[@]}" "${ACC_COMPILERS[@]}" \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON &&
cmake --build "$BUILD_ROOT/openacc" --target "$DIRECTIVE_TARGET" --parallel 2 &&
"$BUILD_ROOT/openacc/nvidia/$SOURCE_LANGUAGE/$LIBRARY/$DIRECTIVE_TARGET"
```


<a id="hip"></a>

## 5. AMD C/C++ HIP

native HIPの`CMAKE_HIP_COMPILER`は対応するClangです。`hipcc` wrapperをここへ指定しません。手動コンパイルでは`hipcc`を使えます。
```bash
test "$VENDOR" = amd || exit 1
: "${ROCM_ROOT:?Set the installed ROCm prefix}"
: "${HIP_CLANGXX:?Set the absolute HIP-capable Clang compiler path, not hipcc}"
: "${AMD_GPU_TARGET:?Set the target supported by your GPU and runtime}"
cmake -S . -B "$BUILD_ROOT/hip" "${COMMON[@]}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_HIP=ON \
  -DGPU_SUITE_BUILD_OPENMP=OFF -DGPU_SUITE_ROCM_ROOT="$ROCM_ROOT" \
  -DCMAKE_HIP_COMPILER="$HIP_CLANGXX" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM_ROOT" \
  -DCMAKE_HIP_ARCHITECTURES="$AMD_GPU_TARGET" &&
cmake --build "$BUILD_ROOT/hip" --target "$GPU_TARGET" --parallel 2 &&
"$BUILD_ROOT/hip/amd/c-cpp/$LIBRARY/$GPU_TARGET"
```


<a id="openmp"></a>

## 6. AMD C/C++ OpenMP target

OpenMPがmapした配列をGPUライブラリへ渡します。USMやcopy省略を仮定しません。rocThrustのOpenMP callerと`rocthrust_wrapper.hip`は別翻訳単位としてリンクされます。
```bash
test "$VENDOR" = amd || exit 1
: "${ROCM_ROOT:?Set the installed ROCm prefix}"
: "${AMD_CLANGXX:?Set the absolute AMD Clang OpenMP C++ compiler path}"
: "${AMD_GPU_TARGET:?Set the target supported by your GPU and runtime}"
WRAPPER_OPTIONS=()
if [ "$LIBRARY" = rocthrust ]; then
  WRAPPER_OPTIONS=("-DCMAKE_HIP_COMPILER=${HIP_CLANGXX:?Set HIP Clang for the wrapper}"
    "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$ROCM_ROOT"
    "-DCMAKE_HIP_ARCHITECTURES=$AMD_GPU_TARGET")
fi
cmake -S . -B "$BUILD_ROOT/openmp" "${COMMON[@]}" "${WRAPPER_OPTIONS[@]}" \
  -DCMAKE_CXX_COMPILER="$AMD_CLANGXX" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_HIP=OFF \
  -DGPU_SUITE_BUILD_OPENMP=ON -DGPU_SUITE_ROCM_ROOT="$ROCM_ROOT" \
  -DGPU_SUITE_AMD_GPU_TARGET="$AMD_GPU_TARGET" &&
cmake --build "$BUILD_ROOT/openmp" --target "$DIRECTIVE_TARGET" --parallel 2 &&
OMP_TARGET_OFFLOAD=MANDATORY \
  "$BUILD_ROOT/openmp/amd/c-cpp/$LIBRARY/$DIRECTIVE_TARGET"
```


<a id="cpu-providers"></a>

## CPU数値ライブラリの選択

CPU依存だけが不足している場合は、CPUを無効にしたGPU手順を使えます。異なるproviderへの自動fallbackはありません。

| 演算 | 選択・必要ファイル |
| --- | --- |
| FFT | FP32 FFTW: `fftw3.h`, `libfftw3f`; Fortran also `fftw3.f03`. `FFTW_ROOT` でprefixを指定。教材は逐次、スレッド版はbenchmark用。 |
| C/C++ BLAS | `GPU_SUITE_CPU_BLAS_BACKEND=ONEMKL`, `OPENBLAS`, `GENERIC_CBLAS` |
| C/C++ Solver | `GPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL`, `OPENBLAS`, `GENERIC_LAPACKE`; LAPACKE API |
| C/C++ Sparse | NVIDIA: `ONEMKL`; AMD: `ONEMKL` / `AOCLSPARSE`. REFERENCEは教材の代替ではありません。 |
| NVIDIA Fortran BLAS / Sparse / Solver | oneMKL LP64, `mkl_rt`, `mkl_service.h`; Sparse: `mkl_spblas.f90`; `MKLROOT` |
| AMD Fortran BLAS / Solver | 明示選択したLP64 Fortran BLAS／LAPACK。AOCLではBLIS／libFLAMEと共通のGENERIC provider設定を使う |
| AMD Fortran Sparse | 明示選択した`AOCLSPARSE`と付属の`aocl_sparse_c.f90` binding |
| RAND / Thrust | C++標準ライブラリ／Fortran組込みだけで逐次CPU教材を実行可能。 |

CMakeの探索範囲外にある場合は、`CMAKE_PREFIX_PATH`や下記の入力を`CPU_OPTIONS`へ追加します。headerとlibraryは同じprovider・ABIから選んでください。AOCL BLIS／libFLAMEはC/C++のGENERICプロバイダで指定でき、並列版を測る場合は意図した並列ライブラリとその依存を選択します。

| Provider | CMake inputs |
| --- | --- |
| Generic CBLAS | `GPU_SUITE_GENERIC_CBLAS_INCLUDE_DIR`, `GPU_SUITE_GENERIC_CBLAS_LIBRARY`, `GPU_SUITE_GENERIC_CBLAS_EXTRA_LIBRARIES` |
| Generic LAPACKE | `GPU_SUITE_GENERIC_LAPACKE_INCLUDE_DIR`, `GPU_SUITE_GENERIC_LAPACKE_LIBRARY`, `GPU_SUITE_GENERIC_LAPACKE_EXTRA_LIBRARIES` |
| AOCL Sparse | `GPU_SUITE_AOCL_SPARSE_INCLUDE_DIR`, `GPU_SUITE_AOCL_SPARSE_LIBRARY`, `GPU_SUITE_AOCL_SPARSE_EXTRA_LIBRARIES` |

`*_EXTRA_LIBRARIES`はリンク順の絶対ファイルパスをセミコロンで区切ります。フラグ文字列ではありません。実行時にも同じ共有ライブラリを解決できるように、インストール環境のloader設定を維持してください。FFTWのスレッド探索は`FFTW3f_THREADING=AUTO/PTHREAD/OPENMP/COMBINED/NONE`で制御できます。これは逐次教材を並列化する設定ではありません。

<a id="check"></a>

## 成功確認とよくある失敗

- configureに成功しても、対象targetが依存不足で無効な場合があります。対象のbuild成功と終了値0、各READMEの期待値を確認します。Fortran教材は`verification PASS`を表示します。
- `target not found`：configureが表示した無効化理由を確認し、必要なheader・API・libraryを揃えます。rocSPARSEはv2 APIが必要です。
- 実行時の共有ライブラリ不足：build時と同じprovider・toolchain環境を使います。別のlibraryへ差し替えて誤魔化さないでください。
- OpenMP host fallback／device mappingエラー：offloadコンパイラとruntime、GPU target、GPU可視性を確認します。CPU実行をGPU成功とは扱いません。
- 数値検証失敗：`info`とAPI戻り値、配列形式、初期値、精度を確認します。条件を勝手に緩めないでください。

この手順は教材だけを構築します。`_bench`も必要な場合は**新しい**build treeで`GPU_SUITE_BUILD_BENCHMARKS=ON`とし、[測定案内](PORTABILITY.ja.md#measuring-on-your-own-system)へ進んでください。ソースが同じでも、別環境でのGPU実行が検証済みになるわけではありません。
