# rocSPARSE C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

CSR形式のFP64疎行列ベクトル積`y = alpha*A*x + beta*y`を計算します。教材行列は規則的な2次元5点Poisson stencilです。

<a id="teaching-default"></a>

## 問題と期待する結果

1024×1024格子から1,048,576行の行列を生成します。対角4、隣接−1、`alpha=beta=1`、`x=y=1`です。出力yは内部1・辺2・角3となります。CSR添字は0始まりです。

終了値0と上記の結果確認が成功条件です。教材のサイズはソース内定数であり、benchmark用CLIを教材へ渡しません。
GPUは`rocsparse_v2_spmv`＋`rocsparse_spmv_alg_csr_rowsplit`を使用します。ROCm 7.2で確認したv2 APIが必要で、旧APIへfallbackしません。analysis用とcompute用のworkspaceを別々に照会します。

<a id="dependencies"></a>

## 必要環境

CPUはAOCL Sparse（`aoclsparse.h`・`libaoclsparse`、LP64）またはoneMKL Sparseを明示選択します。以下はAOCL Sparseです。REFERENCEは補助benchmark用で、教材providerを代替しません。

HIPにはROCmの対応ライブラリ、OpenMP targetには対応するAMD Clangも必要です。native CMake HIPでは`hipcc`ではなくHIP対応Clangを指定します。CPUだけの利用ではROCmは不要です。
下記CPU設定はAOCL Sparseを明示選択します。oneMKLを使う場合だけ`CPU_OPTIONS`を`(-DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL)`へ変更します。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocsparse
CPU_TARGET=sparse_cpu
GPU_TARGET=sparse_gpu
DIRECTIVE_TARGET=openmp_rocsparse
CPU_OPTIONS=(-DGPU_SUITE_CPU_SPARSE_BACKEND=AOCLSPARSE)
```

<a id="canonical-sources-and-targets"></a>

## ソースとtarget

| 方式 | 教材ソース／target | benchmarkソース／target |
| --- | --- | --- |
| CPU | [`sparse_cpu.c`](examples/sparse_cpu.c) / `sparse_cpu` | [`sparse_cpu_bench.c`](benchmarks/sparse_cpu_bench.c) / `sparse_cpu_bench` |
| HIP | [`sparse_gpu.cpp`](examples/sparse_gpu.cpp) / `sparse_gpu` | [`sparse_gpu_bench.cpp`](benchmarks/sparse_gpu_bench.cpp) / `sparse_gpu_bench` |
| OpenMP target | [`openmp_rocsparse.cpp`](examples/openmp_rocsparse.cpp) / `openmp_rocsparse` | [`openmp_rocsparse_bench.cpp`](benchmarks/openmp_rocsparse_bench.cpp) / `openmp_rocsparse_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

CSRのrow pointer・column index・値を一組として置き換え、添字幅・基準・次元・非零要素の順序を一致させます。betaが0でなければ初期yも必要です。descriptorとanalysisは保持する行列・pointerについてAPIが許す範囲で再利用します。規則的な5点行列の性能を不規則な実行列へ一般化しないでください。

HIP版では`hipMalloc`／`hipMemcpy`で配列を管理します。OpenMP target版では`map`が領域を所有し、`use_device_ptr`でGPUライブラリへ渡します。結果取得・map終了より前に完了を保証し、map領域をHIPで解放しないでください。物理メモリを共有するGPUでも、USMやcopyの省略を仮定しません。

<a id="direct-compile"></a>

## 手動コンパイル（任意）

以下はリポジトリ直下のBashで、必要なheader・libraryの探索パスが設定済みの環境を想定します。複雑なCPU依存やstatic linkでは、上のCMake手順を推奨します。CPU・HIP・OpenMPはそれぞれ必要なblockだけを実行してください。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```
AOCLの追加依存が必要なら、`AOCL_EXTRA_LIBRARIES`をリンク順の絶対パスからなるBash配列として設定します。不要なら空配列です。

```bash
AOCL_EXTRA_LIBRARIES=()
```

```bash
cc -std=c17 -O2 -DGPU_SUITE_USE_AOCL_SPARSE=1 \
  -I"${AOCL_SPARSE_INCLUDE:?Set the aoclsparse.h directory}" \
  amd/c-cpp/rocsparse/examples/sparse_cpu.c \
  "${AOCL_SPARSE_LIBRARY:?Set the AOCL Sparse shared library file}" \
  "${AOCL_EXTRA_LIBRARIES[@]}" -lm -o "$DIRECT_BUILD/sparse_cpu" &&
"$DIRECT_BUILD/sparse_cpu"
```
GPUのtargetを実機に合わせて設定し、選択したROCmの`hipcc`を使います。

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocsparse/examples/sparse_gpu.cpp -lrocsparse \
  -o "$DIRECT_BUILD/sparse_gpu" &&
"$DIRECT_BUILD/sparse_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocsparse/examples/openmp_rocsparse.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocsparse -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocsparse" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocsparse"
```

<a id="timing-scopes"></a>

## 性能を測る場合

図ではlibrary／end-to-endと表示し、内部CLI名は`compute`／`end-to-end`を使います。教材全体を囲む単純な時間とは異なります。

libraryはdescriptorとcompute workspaceを再利用し、trial前にyを復元してv2 SpMV computeの反復と完了を測ります。end-to-endはCPUのdescriptor・hint・optimize、GPUのcopy／map・descriptor・v2 analysis・workspace準備、計算・y取得を含みます。analysis bufferの解放は準備の一部です。最後の明示cleanupは計測外ですが、OpenMPの構造化map終了は計測内です。旧adaptive結果とv2／csr_rowsplit結果を混在させません。

host入力の生成と数値検証は計測外です。詳細は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)、全系列の実行・集計は[Sirius手順](../../../jobs/sirius/README.md)を参照してください。配布configは読者の実行環境へ自動適合するものではありません。

<a id="validation-and-limits"></a>

## 確認済みの範囲と制約

Siriusのv2／csr_rowsplit版`a974467`で6成果物のbuild、教材3本、smoke 96件、本測定1,080行が成功しています。campaignは`sirius-rocsparse-v2-20261019T010457Z-a974467`です。旧adaptive結果とは分けて保持しています。

要求CPUスレッド数と実効worker数を同一視しません。RAND／Thrustの保存済みCPU評価は逐次です。完全な実ロードライブラリ同一性も未確認事項として残しています。記録された成功は別環境や後続変更の保証ではなく、ローカルfixture検査はGPU実行の証拠ではありません。

[問題と命名の仕様](../../../docs/PROJECT_SPECIFICATION.md)・[結果形式](../../../docs/RESULT_SCHEMA.md)
