# rocFFT C/C++：ビルド・実行・使い方

[English](README.md) | [日本語](README.ja.md)

長さ1,024のFP32複素数FFTを4,096本まとめてforward変換します。多数の信号をまとめて処理するアプリケーションで、batched FFTの呼び出し方を学べます。

<a id="teaching-default"></a>

## 問題と期待する結果

out-of-placeの1D C2Cです。複素数をinterleaved形式で連続配置し、入力をすべて`1+0i`にします。各FFTのDC成分は`1024+0i`、他は0で、ソース内の許容誤差で確認します。

終了値0と上記の結果確認が成功条件です。教材のサイズはソース内定数であり、benchmark用CLIを教材へ渡しません。

<a id="dependencies"></a>

## 必要環境

CPUには単精度FFTWの`fftw3.h`・`libfftw3f`が必要です。通常の探索範囲外なら`FFTW_ROOT`でprefixを指定します。スレッド版はbenchmark用の任意依存で、逐次教材には不要です。

HIPにはROCmの対応ライブラリ、OpenMP targetには対応するAMD Clangも必要です。native CMake HIPでは`hipcc`ではなくHIP対応Clangを指定します。CPUだけの利用ではROCmは不要です。

<a id="select-a-minimal-build"></a>

## 最小ビルドの選択

Bashで以下を設定してから、[共通ビルド手順](../../../docs/BUILDING.ja.md#select)のCPU・GPU直接版・ディレクティブ版のうち必要な節だけを実行します。教材targetだけを構築し、benchmarkや全件テストは不要です。
```bash
VENDOR=amd
SOURCE_LANGUAGE=c-cpp
LIBRARY=rocfft
CPU_TARGET=fft_cpu
GPU_TARGET=fft_gpu
DIRECTIVE_TARGET=openmp_rocfft
CPU_OPTIONS=()
```

<a id="canonical-sources-and-targets"></a>

## ソースとtarget

| 方式 | 教材ソース／target | benchmarkソース／target |
| --- | --- | --- |
| CPU | [`fft_cpu.c`](examples/fft_cpu.c) / `fft_cpu` | [`fft_cpu_bench.c`](benchmarks/fft_cpu_bench.c) / `fft_cpu_bench` |
| HIP | [`fft_gpu.cpp`](examples/fft_gpu.cpp) / `fft_gpu` | [`fft_gpu_bench.cpp`](benchmarks/fft_gpu_bench.cpp) / `fft_gpu_bench` |
| OpenMP target | [`openmp_rocfft.cpp`](examples/openmp_rocfft.cpp) / `openmp_rocfft` | [`openmp_rocfft_bench.cpp`](benchmarks/openmp_rocfft_bench.cpp) / `openmp_rocfft_bench` |

<a id="adapting-the-example"></a>

## 自分のアプリケーションへ取り込む

精度・変換方向・stride・distance・batchを実際のbufferに合わせます。planは形状と配置が合う間だけ再利用します。FFTW教材は逐次の`FFTW_ESTIMATE`で、スレッド版benchmarkとは別です。rocFFTの入力を再利用する場合は元データを保持し、実行後も入力が不変とは仮定しないでください。

HIP版では`hipMalloc`／`hipMemcpy`で配列を管理します。OpenMP target版では`map`が領域を所有し、`use_device_ptr`でGPUライブラリへ渡します。結果取得・map終了より前に完了を保証し、map領域をHIPで解放しないでください。物理メモリを共有するGPUでも、USMやcopyの省略を仮定しません。

<a id="direct-compile"></a>

## 手動コンパイル（任意）

以下はリポジトリ直下のBashで、必要なheader・libraryの探索パスが設定済みの環境を想定します。複雑なCPU依存やstatic linkでは、上のCMake手順を推奨します。CPU・HIP・OpenMPはそれぞれ必要なblockだけを実行してください。

```bash
DIRECT_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-direct.XXXXXX")" || exit 1
```

```bash
cc -std=c17 -O2 amd/c-cpp/rocfft/examples/fft_cpu.c -lfftw3f -lm -o "$DIRECT_BUILD/fft_cpu" &&
"$DIRECT_BUILD/fft_cpu"
```
GPUのtargetを実機に合わせて設定し、選択したROCmの`hipcc`を使います。

```bash
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
hipcc -x hip -std=c++17 -O2 --offload-arch="$AMD_GPU_TARGET" \
  amd/c-cpp/rocfft/examples/fft_gpu.cpp -lrocfft \
  -o "$DIRECT_BUILD/fft_gpu" &&
"$DIRECT_BUILD/fft_gpu"
```

```bash
: "${AMD_CLANGXX:?Set the AMD Clang OpenMP C++ compiler}"
: "${ROCM_ROOT:?Set the ROCm prefix containing include}"
: "${ROCM_LIBDIR:?Set the ROCm shared-library directory}"
: "${AMD_GPU_TARGET:?Set your AMD GPU target}"
"$AMD_CLANGXX" -std=c++17 -O2 -fopenmp --offload-arch="$AMD_GPU_TARGET" \
  -I"$ROCM_ROOT/include" amd/c-cpp/rocfft/examples/openmp_rocfft.cpp \
  -L"$ROCM_LIBDIR" -Wl,-rpath,"$ROCM_LIBDIR" -lrocfft -lamdhip64 \
  -o "$DIRECT_BUILD/openmp_rocfft" &&
OMP_TARGET_OFFLOAD=MANDATORY "$DIRECT_BUILD/openmp_rocfft"
```

<a id="timing-scopes"></a>

## 性能を測る場合

図ではlibrary／end-to-endと表示し、内部CLI名は`compute`／`end-to-end`を使います。教材全体を囲む単純な時間とは異なります。

libraryではplan・execution info・workspace・device／map配列を再利用します。必要な入力復元を各FFTの計測外で行い、FFTと完了待ちの個別計時を加算してrepeatで割ります。NVIDIA FFTのrepeat列の計時とは異なります。end-to-endはplan／workspaceの準備、device確保／map、入力移動、FFT・完了・結果取得を含み、HIPは出力ゼロ初期化も含みます。OpenMPのmap終了は計測内、明示的なplan／workspaceとHIP配列の後処理は計測外です。AMD CPU／GPU比較では承認済みの現行方式を維持し、NVIDIAとの完全同一計時とは扱いません。

host入力の生成と数値検証は計測外です。詳細は[測定仕様](../../../docs/BENCHMARK_PROTOCOL.ja.md)、全系列の実行・集計は[Sirius手順](../../../jobs/sirius/README.md)を参照してください。配布configは読者の実行環境へ自動適合するものではありません。

<a id="validation-and-limits"></a>

## 確認済みの範囲と制約

SiriusでCPU／HIP／OpenMPのbuild・数値検証と性能測定の保存記録があります。版・試験範囲は[検証報告](../../../docs/VALIDATION_REPORT.md#amd-production-comparison)に記録しています。

要求CPUスレッド数と実効worker数を同一視しません。RAND／Thrustの保存済みCPU評価は逐次です。完全な実ロードライブラリ同一性も未確認事項として残しています。記録された成功は別環境や後続変更の保証ではなく、ローカルfixture検査はGPU実行の証拠ではありません。

[問題と命名の仕様](../../../docs/PROJECT_SPECIFICATION.md)・[結果形式](../../../docs/RESULT_SCHEMA.md)
