# rocBLAS C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

FP64 DGEMMで密行列積`C = alpha*A*B + beta*C`を計算します。CPU BLASからGPU直接呼び出し・ディレクティブによる配列管理への対応を確認できます。

<a id="teaching-default"></a>

## 問題と期待する結果

列優先の正方行列で`m=n=k=1024`、`alpha=beta=1`、A/B/Cはすべて1です。出力は全要素`1025`となり、C/C++教材は最大誤差`1e-10`以下で成功します。

終了値0と上記の結果確認が成功条件です。教材のサイズはソース内定数であり、benchmark用CLIを教材へ渡しません。
CPUはCBLAS、GPUは`rocblas_dgemm`を呼びます。

<a id="dependencies"></a>

## 必要環境

CPUではoneMKL・OpenBLAS・generic CBLASのいずれかを明示選択します。以下の例はOpenBLAS（`cblas.h`・`libopenblas`）を選びます。

HIPにはROCmの対応ライブラリ、OpenMP targetには対応するAMD Clangも必要です。native CMake HIPでは`hipcc`ではなくHIP対応Clangを指定します。CPUだけの利用ではROCmは不要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocblas
CPU_TARGET=blas_cpu
GPU_TARGET=blas_gpu
DIRECTIVE_TARGET=openmp_rocblas
CPU_OPTIONS=(-DGPU_SUITE_CPU_BLAS_BACKEND=OPENBLAS)
```

<a id="canonical-sources-and-targets"></a>

## ソースとtarget

| 方式 | 教材ソース／target | benchmarkソース／target |
| --- | --- | --- |
| CPU | [`blas_cpu.c`](examples/blas_cpu.c) / `blas_cpu` | [`blas_cpu_bench.c`](benchmarks/blas_cpu_bench.c) / `blas_cpu_bench` |
| HIP | [`blas_gpu.cpp`](examples/blas_gpu.cpp) / `blas_gpu` | [`blas_gpu_bench.cpp`](benchmarks/blas_gpu_bench.cpp) / `blas_gpu_bench` |
| OpenMP target | [`openmp_rocblas.cpp`](examples/openmp_rocblas.cpp) / `openmp_rocblas` | [`openmp_rocblas_bench.cpp`](benchmarks/openmp_rocblas_bench.cpp) / `openmp_rocblas_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

配列の格納順とleading dimension（`lda`、`ldb`、`ldc`）を合わせます。betaが0でないためCも入力です。A/Bだけをコピーしてはいけません。繰り返す場合は配列とhandleを適切な寿命で保持します。CPUの並列性は選択したBLAS providerが担い、要求スレッド数は実測worker数ではありません。

HIP版では`hipMalloc`／`hipMemcpy`で配列を管理します。OpenMP target版では`map`が領域を所有し、`use_device_ptr`でGPUライブラリへ渡します。結果取得・map終了より前に完了を保証し、map領域をHIPで解放しないでください。物理メモリを共有するGPUでも、USMやcopyの省略を仮定しません。

<a id="direct-compile"></a>

## 手動コンパイル（任意）

以下はリポジトリ直下のBashで、必要なheader・libraryの探索パスが設定済みの環境を想定します。複雑なCPU依存やstatic linkでは、上のCMake手順を推奨します。CPU・HIP・OpenMPはそれぞれ必要なblockだけを実行してください。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 amd/c-cpp/rocblas/examples/blas_cpu.c -lopenblas -lm -o "$DIRECT_BUILD/blas_cpu" &&
"$DIRECT_BUILD/blas_cpu"
```
GPUのtargetを実機に合わせて設定し、選択したROCmの`hipcc`を使います。

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocblas/examples/blas_gpu.cpp -lrocblas \
  -o "$DIRECT_BUILD/blas_gpu" &&
"$DIRECT_BUILD/blas_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocblas/examples/openmp_rocblas.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocblas -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocblas" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocblas"
```

<a id="timing-scopes"></a>

## 性能を測る場合

図ではlibrary／end-to-endと表示し、内部CLI名は`compute`／`end-to-end`を使います。教材全体を囲む単純な時間とは異なります。

libraryはCを計測外で復元し、保持した配列・handleでDGEMMの連続更新と完了を測ります。end-to-endはhost Cの復元を計測外で行い、device確保／map、入力移動、handle作成、DGEMM・完了・C取得を含めます。明示的なHIP解放とhandle破棄は計測外、OpenMPの構造化map終了は計測内です。beta=1のlibrary検証はrepeat回すべての更新を反映します。

host入力の生成と数値検証は計測外です。詳細は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)、全系列の実行・集計は[Sirius手順](../../../jobs/sirius/README.md)を参照してください。配布configは読者の実行環境へ自動適合するものではありません。

<a id="validation-and-limits"></a>

## 確認済みの範囲と制約

SiriusでCPU／HIP／OpenMPのbuild・数値検証と性能測定の保存記録があります。版・試験範囲は[検証報告](../../../docs/VALIDATION_REPORT.md#amd-production-comparison)に記録しています。

要求CPUスレッド数と実効worker数を同一視しません。RAND／Thrustの保存済みCPU評価は逐次です。完全な実ロードライブラリ同一性も未確認事項として残しています。記録された成功は別環境や後続変更の保証ではなく、ローカルfixture検査はGPU実行の証拠ではありません。

[問題と命名の仕様](../../../docs/PROJECT_SPECIFICATION.md)・[結果形式](../../../docs/RESULT_SCHEMA.md)
