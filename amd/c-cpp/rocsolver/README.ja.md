# rocSOLVER C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

LU分解を使って複数右辺のFP64密行列方程式`A*X=B`を解きます。行列のほかpivot配列とsolverの終了情報を扱う流れを示します。

<a id="teaching-default"></a>

## 問題と期待する結果

`N=1024`、`nrhs=16`、`A=N*I+ones`、Bの全要素は`2*N`です。解はすべて1となります。APIの成功と`info=0`に加え、数値検証の成功が必要です。

終了値0と上記の結果確認が成功条件です。教材のサイズはソース内定数であり、benchmark用CLIを教材へ渡しません。

<a id="dependencies"></a>

## 必要環境

CPUはLAPACKE APIが必要で、oneMKL・LAPACKEを含むOpenBLAS・generic LAPACKEを明示選択します。以下の例はgeneric LAPACKEです。BLASだけの環境では不足します。

HIPにはROCmの対応ライブラリ、OpenMP targetには対応するAMD Clangも必要です。native CMake HIPでは`hipcc`ではなくHIP対応Clangを指定します。CPUだけの利用ではROCmは不要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocsolver
CPU_TARGET=solver_cpu
GPU_TARGET=solver_gpu
DIRECTIVE_TARGET=openmp_rocsolver
CPU_OPTIONS=(-DGPU_SUITE_CPU_LAPACK_BACKEND=GENERIC_LAPACKE)
```

<a id="canonical-sources-and-targets"></a>

## ソースとtarget

| 方式 | 教材ソース／target | benchmarkソース／target |
| --- | --- | --- |
| CPU | [`solver_cpu.c`](examples/solver_cpu.c) / `solver_cpu` | [`solver_cpu_bench.c`](benchmarks/solver_cpu_bench.c) / `solver_cpu_bench` |
| HIP | [`solver_gpu.cpp`](examples/solver_gpu.cpp) / `solver_gpu` | [`solver_gpu_bench.cpp`](benchmarks/solver_gpu_bench.cpp) / `solver_gpu_bench` |
| OpenMP target | [`openmp_rocsolver.cpp`](examples/openmp_rocsolver.cpp) / `openmp_rocsolver` | [`openmp_rocsolver_bench.cpp`](benchmarks/openmp_rocsolver_bench.cpp) / `openmp_rocsolver_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

分解でA、solveでBが上書きされるため、残差確認や再試行に必要な原本を保持します。API失敗とinfoによる特異性の報告を区別してください。Aが同じでBだけ変わるなら、LUとpivotを保持してgetrsへ渡せます。毎回分解する場合とは費用が異なります。

HIP版では`hipMalloc`／`hipMemcpy`で配列を管理します。OpenMP target版では`map`が領域を所有し、`use_device_ptr`でGPUライブラリへ渡します。結果取得・map終了より前に完了を保証し、map領域をHIPで解放しないでください。物理メモリを共有するGPUでも、USMやcopyの省略を仮定しません。

<a id="direct-compile"></a>

## 手動コンパイル（任意）

以下はリポジトリ直下のBashで、必要なheader・libraryの探索パスが設定済みの環境を想定します。複雑なCPU依存やstatic linkでは、上のCMake手順を推奨します。CPU・HIP・OpenMPはそれぞれ必要なblockだけを実行してください。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 amd/c-cpp/rocsolver/examples/solver_cpu.c \
  -llapacke -llapack -lblas -lm -o "$DIRECT_BUILD/solver_cpu" &&
"$DIRECT_BUILD/solver_cpu"
```
GPUのtargetを実機に合わせて設定し、選択したROCmの`hipcc`を使います。

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocsolver/examples/solver_gpu.cpp -lrocsolver -lrocblas \
  -o "$DIRECT_BUILD/solver_gpu" &&
"$DIRECT_BUILD/solver_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocsolver/examples/openmp_rocsolver.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocsolver -lrocblas -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocsolver" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocsolver"
```

<a id="timing-scopes"></a>

## 性能を測る場合

図ではlibrary／end-to-endと表示し、内部CLI名は`compute`／`end-to-end`を使います。教材全体を囲む単純な時間とは異なります。

libraryはA/Bを計測外で復元し、getrf・完了・info取得確認、続いてgetrs・完了を含みます。repeatは1です。end-to-endではhandle／workspace・配列準備、復元copy、最終解の取得も含みます。明示的なOpenMP data削除を含むcleanupは計測外です。CPU教材はdgesv、benchmarkはgetrf／getrsへ分割します。残差検証用のA/B原本を保持します。

host入力の生成と数値検証は計測外です。詳細は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)、全系列の実行・集計は[Sirius手順](../../../jobs/sirius/README.md)を参照してください。配布configは読者の実行環境へ自動適合するものではありません。

<a id="validation-and-limits"></a>

## 確認済みの範囲と制約

SiriusでCPU／HIP／OpenMPのbuild・数値検証と性能測定の保存記録があります。版・試験範囲は[検証報告](../../../docs/VALIDATION_REPORT.md#amd-production-comparison)に記録しています。

要求CPUスレッド数と実効worker数を同一視しません。RAND／Thrustの保存済みCPU評価は逐次です。完全な実ロードライブラリ同一性も未確認事項として残しています。記録された成功は別環境や後続変更の保証ではなく、ローカルfixture検査はGPU実行の証拠ではありません。

[問題と命名の仕様](../../../docs/PROJECT_SPECIFICATION.md)・[結果形式](../../../docs/RESULT_SCHEMA.md)
