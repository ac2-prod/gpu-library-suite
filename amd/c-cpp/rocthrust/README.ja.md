# rocThrust C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

transform-reduceで`sum(values[i]^2)`を計算します。変換と集約を組み合わせ、二乗値の中間配列を保存せずに求めます。

<a id="teaching-default"></a>

## 問題と期待する結果

入力は`2^24`個のFP64の1で、二乗和の期待値は`16777216`です。CPUは逐次STLの`std::transform_reduce`です。hostへ返す結果はdouble 1個です。

終了値0と上記の結果確認が成功条件です。教材のサイズはソース内定数であり、benchmark用CLIを教材へ渡しません。

<a id="dependencies"></a>

## 必要環境

逐次CPUはC++17と標準ライブラリだけで動きます。GPUにはrocThrust／rocPRIMとHIPが必要です。

HIPにはROCmの対応ライブラリ、OpenMP targetには対応するAMD Clangも必要です。native CMake HIPでは`hipcc`ではなくHIP対応Clangを指定します。CPUだけの利用ではROCmは不要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocthrust
CPU_TARGET=reduce_cpu
GPU_TARGET=reduce_gpu
DIRECTIVE_TARGET=openmp_rocthrust
CPU_OPTIONS=(-DGPU_SUITE_CPU_THRUST_BACKEND=STL)
```

<a id="canonical-sources-and-targets"></a>

## ソースとtarget

| 方式 | 教材ソース／target | benchmarkソース／target |
| --- | --- | --- |
| CPU | [`reduce_cpu.cpp`](examples/reduce_cpu.cpp) / `reduce_cpu` | [`reduce_cpu_bench.cpp`](benchmarks/reduce_cpu_bench.cpp) / `reduce_cpu_bench` |
| HIP | [`reduce_gpu.cpp`](examples/reduce_gpu.cpp) / `reduce_gpu` | [`reduce_gpu_bench.cpp`](benchmarks/reduce_gpu_bench.cpp) / `reduce_gpu_bench` |
| OpenMP target | [`openmp_rocthrust.cpp`](examples/openmp_rocthrust.cpp) / `openmp_rocthrust` | [`openmp_rocthrust_bench.cpp`](benchmarks/openmp_rocthrust_bench.cpp) / `openmp_rocthrust_bench` |

OpenMP教材・benchmarkは[rocthrust_wrapper.hip](examples/rocthrust_wrapper.hip)も明示的にcompile・linkします。C ABIは`int rocthrust_square_sum(const double*, size_t, double*) noexcept`で、戻り値は状態、結果は第3引数へ書きます。NVIDIA Fortranのwrapperとは関数仕様が異なります。wrapperはcallerのmapを保持・解放せず、例外もC ABI外へ出しません。

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

変換関数・集約演算・単位元を一組で変更します。浮動小数点の集約は順序で結果が変わり得るため、一般入力では適切な誤差許容値を使います。繰り返すGPU処理では入力を常駐させます。ディレクティブが管理するpointerをwrapperで解放せず、確保した側に所有権を残します。

HIP直接教材は`thrust::device_vector<double>`をdevice側で確保・初期化し、host入力配列を`hipMemcpy`で明示コピーする構成ではありません。OpenMP target版では`map`が領域を所有し、`use_device_ptr`でGPUライブラリへ渡します。結果取得・map終了より前に完了を保証し、map領域をHIPで解放しないでください。物理メモリを共有するGPUでも、USMやcopyの省略を仮定しません。

<a id="direct-compile"></a>

## 手動コンパイル（任意）

以下はリポジトリ直下のBashで、必要なheader・libraryの探索パスが設定済みの環境を想定します。複雑なCPU依存やstatic linkでは、上のCMake手順を推奨します。CPU・HIP・OpenMPはそれぞれ必要なblockだけを実行してください。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 amd/c-cpp/rocthrust/examples/reduce_cpu.cpp -o "$DIRECT_BUILD/reduce_cpu" &&
"$DIRECT_BUILD/reduce_cpu"
```
GPUのtargetを実機に合わせて設定し、選択したROCmの`hipcc`を使います。

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocthrust/examples/reduce_gpu.cpp \
  -o "$DIRECT_BUILD/reduce_gpu" &&
"$DIRECT_BUILD/reduce_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  -c amd/c-cpp/rocthrust/examples/rocthrust_wrapper.hip \
  -o "$DIRECT_BUILD/rocthrust_wrapper.o" &&
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" -c amd/c-cpp/rocthrust/examples/openmp_rocthrust.cpp \
  -o "$DIRECT_BUILD/openmp_rocthrust.o" &&
"$AMD_CLANGXX" -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  "$DIRECT_BUILD/openmp_rocthrust.o" "$DIRECT_BUILD/rocthrust_wrapper.o" \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocthrust" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocthrust"
```

<a id="timing-scopes"></a>

## 性能を測る場合

図ではlibrary／end-to-endと表示し、内部CLI名は`compute`／`end-to-end`を使います。教材全体を囲む単純な時間とは異なります。

libraryはGPU入力を保持し、同期的なtransform_reduce呼び出しを反復計測します。end-to-endは確保／map・入力copy、集約とhost scalar結果取得を含み、明示的な解放／data削除は計測外です。入力は不変で、反復ごとの復元は不要です。各集約は単位元0.0から始めます。保存済み本評価のCPU系列は逐次STLです。

host入力の生成と数値検証は計測外です。詳細は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)、全系列の実行・集計は[Sirius手順](../../../jobs/sirius/README.md)を参照してください。配布configは読者の実行環境へ自動適合するものではありません。

<a id="validation-and-limits"></a>

## 確認済みの範囲と制約

SiriusでCPU／HIP／OpenMPのbuild・数値検証と性能測定の保存記録があります。版・試験範囲は[検証報告](../../../docs/VALIDATION_REPORT.md#amd-production-comparison)に記録しています。

要求CPUスレッド数と実効worker数を同一視しません。RAND／Thrustの保存済みCPU評価は逐次です。完全な実ロードライブラリ同一性も未確認事項として残しています。記録された成功は別環境や後続変更の保証ではなく、ローカルfixture検査はGPU実行の証拠ではありません。

[問題と命名の仕様](../../../docs/PROJECT_SPECIFICATION.md)・[結果形式](../../../docs/RESULT_SCHEMA.md)
