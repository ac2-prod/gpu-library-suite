# rocFFT C/C++ 教材とベンチマーク

[English](README.md) | [日本語](README.ja.md)

FFTW3f CPU、HIP/rocFFT、OpenMP target/rocFFTを比較します。記録済みのSirius
baselineでは3系列のbuild、教材3本、CPU単スレッド・2スレッドを含む96件の
基本smokeが成功しました。[証跡と範囲](../../../docs/VALIDATION_REPORT.md#amd-rocfft-sirius-baseline)
を参照してください。後続変更や残る5ライブラリには別の検証が必要です。
小規模な数値smokeから性能・scalingの成功を主張しません。

## ソースとターゲット

| 実装 | 教材ソース／ターゲット | ベンチマークソース／ターゲット |
| --- | --- | --- |
| CPU | `examples/fft_cpu.c`／`fft_cpu` | `benchmarks/fft_cpu_bench.c`／`fft_cpu_bench` |
| HIP | `examples/fft_gpu.cpp`／`fft_gpu` | `benchmarks/fft_gpu_bench.cpp`／`fft_gpu_bench` |
| OpenMP target | `examples/openmp_rocfft.cpp`／`openmp_rocfft` | `benchmarks/openmp_rocfft_bench.cpp`／`openmp_rocfft_bench` |

教材は各1ソースで完結し、ライブラリ呼出しを直接示します。
FP32 interleaved 複素数、out-of-place batched 1D C2C forward、
`nfft=1024`、`batch=4096`、全入力 `1+0i` です。
CPU は `fftwf_plan_many_dft` を使います。OpenMP は `target data` の
`map(to:)`／`map(from:)` と `use_device_ptr` を使い、rocFFT の完了を待って
データ領域を終了します。仕様の正本は
[PROJECT_SPECIFICATION](../../../docs/PROJECT_SPECIFICATION.md#rocfft-cc-inventory-and-teaching-workload) です。

## 依存とビルド

CPU は FFTW3f、threaded backend は追加で FFTW3f threads API/library を
必要とします。CPU-only では ROCm は不要です。HIP は対応 Clang、HIP Runtime、
rocFFT、OpenMP は AMD 側の target 対応コンパイラと同ライブラリを必要とします。
実 API の compile/link 検査で判定し、依存不足の target だけを理由付きで無効化します。

CPU／HIP／OpenMP は新しい別 build tree とし、共通に
`GPU_SUITE_VENDOR=amd`、`GPU_SUITE_SOURCE_LANGUAGE=c-cpp`、
`GPU_SUITE_BUILD_CUDA=OFF`、`GPU_SUITE_BUILD_OPENACC=OFF` を指定します。
各 tree で `GPU_SUITE_BUILD_CPU`／`GPU_SUITE_BUILD_HIP`／
`GPU_SUITE_BUILD_OPENMP` の該当1種類だけを `ON` にします。

- CPU：`CMAKE_C_COMPILER`、`CMAKE_CXX_COMPILER`、`FFTW_ROOT`。
- HIP：対応 Clang の `CMAKE_HIP_COMPILER`、明示的な
  `CMAKE_HIP_ARCHITECTURES`、`GPU_SUITE_ROCM_ROOT`。
- OpenMP：AMD Clang の `CMAKE_CXX_COMPILER`、
  `GPU_SUITE_AMD_GPU_TARGET`、`GPU_SUITE_ROCM_ROOT`。

教材の手動ビルドでは `hipcc` を使用できますが、native CMake HIP の
`CMAKE_HIP_COMPILER` に hipcc wrapper を固定しません。HIP の `.cpp` は
明示的に HIP 言語としてコンパイルします。未確認の compiler/library path、
GPU target、`native` 指定、thread 数は採用しません。コマンド例は
[英語版](README.md#dependencies-and-build-separation) を参照してください。

Sirius の実行は別途承認された計算ノード割当て後に限ります。
ログインノードで build/GPU 実行は行いません。準備済みの
[build/smoke 本体とサイト入力](../../../jobs/sirius/README.md#next-rocfft-build-and-smoke-compute-node-only)
は今回は実行していません。

## 計測と検証

既存 CLI の `--size`（FFT 長）、`--batch`、`--scope`、`--warmup`、
`--repeat`、`--trials`、許容誤差指定と共通 timer/result writer を使います。
CPU は `cpu-fftw-serial`／`cpu-fftw-threaded` を区別し、threaded が使えない場合に
serial へ無言で置換しません。共通 runner が JSONL/CSV 出力を管理します。

compute は plan、execution info、明示的 workspace、配列を準備します。
rocFFT は out-of-place の入力も変更し得るため、CPU/HIP/OpenMP とも各 repeat 前に
入力・出力を計測外で復元し、FFT と完了同期の各区間を合計します。
wall timestamp は計測外の復元時間を含むことがあります。warm-up も毎回復元します。
`rocfft_setup`／`rocfft_cleanup` は両 scope ともプログラム全体で各1回、計測外です。
end-to-end は各 repeat の plan/info setup、確保、転送、
実行、完了、結果取得を含み、OpenMP の data 領域への入出と `map(from:)`
の完了を落としません。入力生成・検証・明示的後処理は既存の計測外規則に従います。
全 batch の DC／non-DC と非有限値を検証します。詳細は
[BENCHMARK_PROTOCOL](../../../docs/BENCHMARK_PROTOCOL.md#rocfft-cpuhipopenmp-contract) が正本です。

`--device` は HIP の番号、`OMP_DEFAULT_DEVICE` は独立した OpenMP 番号です。
番号一致だけで同じ物理 GPU と判断しません。計測外で実 target 実行と mapped pointer の
HIP 属性を確認し、host fallback を成功と扱いません。実 smoke は
`OMP_TARGET_OFFLOAD=MANDATORY` を設定します。OpenMP 管理配列を HIP で
二重確保・解放せず、USM／stream 相互運用へ置換しません。

HIP runtime 情報は `parameters.runtime` に記録し、CUDA フィールドは null に
保ちます。rocFFT の実 API から library version を取得します。数学的比較キーと
runtime を分けても source/config/manifest/compiler/build 条件/binary/library/runtime
が異なる来歴を同一 campaign に混在させません。
[RESULT_SCHEMA](../../../docs/RESULT_SCHEMA.md#rocfft-additive-representation-and-campaign-identity)
が形式と来歴整合性の正本です。
