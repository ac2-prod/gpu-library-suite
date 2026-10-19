# rocRAND C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

FP64の一様乱数を生成します。シミュレーション入力や確率的アルゴリズムで使うgeneratorと出力bufferの扱いを学べます。

<a id="teaching-default"></a>

## 問題と期待する結果

seed 1234で`2^24`個を生成します。CPUは逐次`std::mt19937_64`、GPUはライブラリ既定の疑似乱数generatorです。CPUの区間は[0,1)、GPUは(0,1]です。同じ分布の比較であり、同じ乱数列ではありません。

終了値0と上記の結果確認が成功条件です。教材のサイズはソース内定数であり、benchmark用CLIを教材へ渡しません。

<a id="dependencies"></a>

## 必要環境

CPUはC++17と標準ライブラリだけで動きます。GPUにはrocRANDが必要です。

HIPにはROCmの対応ライブラリ、OpenMP targetには対応するAMD Clangも必要です。native CMake HIPでは`hipcc`ではなくHIP対応Clangを指定します。CPUだけの利用ではROCmは不要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocrand
CPU_TARGET=rand_cpu
GPU_TARGET=rand_gpu
DIRECTIVE_TARGET=openmp_rocrand
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## ソースとtarget

| 方式 | 教材ソース／target | benchmarkソース／target |
| --- | --- | --- |
| CPU | [`rand_cpu.cpp`](examples/rand_cpu.cpp) / `rand_cpu` | [`rand_cpu_bench.cpp`](benchmarks/rand_cpu_bench.cpp) / `rand_cpu_bench` |
| HIP | [`rand_gpu.cpp`](examples/rand_gpu.cpp) / `rand_gpu` | [`rand_gpu_bench.cpp`](benchmarks/rand_gpu_bench.cpp) / `rand_gpu_bench` |
| OpenMP target | [`openmp_rocrand.cpp`](examples/openmp_rocrand.cpp) / `openmp_rocrand` | [`openmp_rocrand_bench.cpp`](benchmarks/openmp_rocrand_bench.cpp) / `openmp_rocrand_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

用途に合うgeneratorとseed／streamの方針を選びます。同じ列の再生成を意図しない限り、毎回seedを戻しません。継続する列ではgeneratorとbufferを再利用します。CPUとGPUのアルゴリズムは異なるため、要素一致ではなく範囲・分布統計で確認します。CPU教材は逐次です。

HIP版では`hipMalloc`／`hipMemcpy`で配列を管理します。OpenMP target版では`map`が領域を所有し、`use_device_ptr`でGPUライブラリへ渡します。結果取得・map終了より前に完了を保証し、map領域をHIPで解放しないでください。物理メモリを共有するGPUでも、USMやcopyの省略を仮定しません。

<a id="direct-compile"></a>

## 手動コンパイル（任意）

以下はリポジトリ直下のBashで、必要なheader・libraryの探索パスが設定済みの環境を想定します。複雑なCPU依存やstatic linkでは、上のCMake手順を推奨します。CPU・HIP・OpenMPはそれぞれ必要なblockだけを実行してください。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
c++ -std=c++17 -O2 amd/c-cpp/rocrand/examples/rand_cpu.cpp -o "$DIRECT_BUILD/rand_cpu" &&
"$DIRECT_BUILD/rand_cpu"
```
GPUのtargetを実機に合わせて設定し、選択したROCmの`hipcc`を使います。

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocrand/examples/rand_gpu.cpp -lrocrand \
  -o "$DIRECT_BUILD/rand_gpu" &&
"$DIRECT_BUILD/rand_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocrand/examples/openmp_rocrand.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocrand -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocrand" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocrand"
```

<a id="timing-scopes"></a>

## 性能を測る場合

図ではlibrary／end-to-endと表示し、内部CLI名は`compute`／`end-to-end`を使います。教材全体を囲む単純な時間とは異なります。

libraryはgenerator・出力配列を保持し、warm-up後と各trial前に計測外で状態を復元・初期化して、生成の反復と完了を測ります。end-to-endではgenerator作成・初期化・出力確保・生成・完了・全結果取得を含み、generator破棄と領域解放は計測外です。CPU／GPUの乱数アルゴリズムは異なり、乱数列の要素一致やCPU並列実行を意味しません。

host入力の生成と数値検証は計測外です。詳細は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)、全系列の実行・集計は[Sirius手順](../../../jobs/sirius/README.md)を参照してください。配布configは読者の実行環境へ自動適合するものではありません。

<a id="validation-and-limits"></a>

## 確認済みの範囲と制約

SiriusでCPU／HIP／OpenMPのbuild・数値検証と性能測定の保存記録があります。版・試験範囲は[検証報告](../../../docs/VALIDATION_REPORT.md#amd-production-comparison)に記録しています。

要求CPUスレッド数と実効worker数を同一視しません。RAND／Thrustの保存済みCPU評価は逐次です。完全な実ロードライブラリ同一性も未確認事項として残しています。記録された成功は別環境や後続変更の保証ではなく、ローカルfixture検査はGPU実行の証拠ではありません。

[問題と命名の仕様](../../../docs/PROJECT_SPECIFICATION.md)・[結果形式](../../../docs/RESULT_SCHEMA.md)
