# GPU Library Suite

[English](README.md) | [日本語](README.ja.md)

GPUライブラリを自分のアプリケーションへ取り入れるための、CPU版・GPU直接呼び出し版・ディレクティブ版の比較サンプル集です。FFT、行列積、疎行列積、連立一次方程式、乱数生成、リダクションを扱います。

NVIDIA C/C++・NVIDIA Fortran・AMD C/C++を収録しています。各`examples/`は入力生成からライブラリ呼び出し、結果確認までを読める教材です。性能を測りたい場合には、別の`benchmarks/`を使います。

<a id="start-here"></a>

## 使いたい実装を選ぶ

必要なGPUとコンパイラがある列を選び、まず1つの演算を動かしてください。CPUサンプルだけならGPUは不要です。一般のLinux環境で利用でき、Pegasus／Siriusのアカウントは必要ありません。AMD Fortranは未実装です。

| 演算 | NVIDIA C/C++：CUDA・OpenACC | NVIDIA Fortran：CUDA Fortran・OpenACC | AMD C/C++：HIP・OpenMP target |
| --- | --- | --- | --- |
| FFT | [cufft](nvidia/c-cpp/cufft/README.ja.md) | [cufft](nvidia/fortran/cufft/README.ja.md) | [rocfft](amd/c-cpp/rocfft/README.ja.md) |
| BLAS | [cublas](nvidia/c-cpp/cublas/README.ja.md) | [cublas](nvidia/fortran/cublas/README.ja.md) | [rocblas](amd/c-cpp/rocblas/README.ja.md) |
| Sparse | [cusparse](nvidia/c-cpp/cusparse/README.ja.md) | [cusparse](nvidia/fortran/cusparse/README.ja.md) | [rocsparse](amd/c-cpp/rocsparse/README.ja.md) |
| Solver | [cusolver](nvidia/c-cpp/cusolver/README.ja.md) | [cusolver](nvidia/fortran/cusolver/README.ja.md) | [rocsolver](amd/c-cpp/rocsolver/README.ja.md) |
| RAND | [curand](nvidia/c-cpp/curand/README.ja.md) | [curand](nvidia/fortran/curand/README.ja.md) | [rocrand](amd/c-cpp/rocrand/README.ja.md) |
| Thrust | [thrust](nvidia/c-cpp/thrust/README.ja.md) | [thrust](nvidia/fortran/thrust/README.ja.md) | [rocthrust](amd/c-cpp/rocthrust/README.ja.md) |

[共通のビルド・実行手順](docs/BUILDING.ja.md)では、選んだ教材だけを構築します。Fortran固有の条件は[Fortran案内](nvidia/fortran/README.ja.md)を参照してください。

<a id="first-example"></a>

## 最初のCPUサンプル

リポジトリ直下でBashから実行します。C++17コンパイラだけを使い、2²⁴個の1の二乗和を計算します。必要なメモリは入力配列だけで128 MiBです。
```bash
EXAMPLE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu-suite-example.XXXXXX")" || exit 1
c++ -std=c++17 -O2 nvidia/c-cpp/thrust/examples/reduce_cpu.cpp \
  -o "$EXAMPLE_DIR/reduce_cpu" && "$EXAMPLE_DIR/reduce_cpu"
```
正常なら二乗和`16777216`を表示して終了値0となります。コンパイル失敗や非ゼロ終了時は先へ進まず、エラーを確認してください。GPU版・Fortran版は上の表から選びます。共有計算機では、そのシステムの規則に従ってビルド・実行を計算ジョブ内で行ってください。

<a id="using-the-examples"></a>

## 自分のアプリケーションへ取り込む

1. CPUとGPUの`examples/`を並べ、入力形式・精度・演算を確認します。
2. サンプルの入力生成を自分のデータに置き換え、配列の格納順、サイズ、leading dimension、CSRの添字基準を合わせます。
3. 配列の確保・解放をどちらが所有するかを決め、結果取得前の完了を保証します。OpenACC／OpenMP target版も同じGPUライブラリを呼び、主にデータ管理が異なります。
4. 同じ形状で繰り返す場合は、再利用条件を確認して配列・plan・handle・descriptorを保持します。
5. 戻り値、solverの`info`、数値誤差を確認します。サンプルの全要素1に対する期待値は、自分の入力にはそのまま使えません。

各ライブラリのREADMEに具体的な変更箇所と注意点があります。C/C++教材は原則1ファイルです。rocThrust OpenMP版だけは`rocthrust_wrapper.hip`も必要です。Fortranの行列生成helperとThrust wrapperも明示的なコンパイル・リンク入力です。

<a id="requirements-and-cpu-only-build"></a>

## 必要環境

CMake 3.20以上（native HIPとrocThrust OpenMPは3.21以上）、C17／C++17が基本です。Fortran CPUにはGNU FortranまたはNVHPC、CUDA FortranにはNVHPCが必要です。GPU実行には対応GPU・ドライバ・CUDAまたはROCm、ディレクティブ版には対応するoffloadコンパイラが必要です。

CPUの追加ライブラリは演算ごとに異なります。RAND／Thrustの逐次CPU例には外部数値ライブラリが不要です。依存不足では該当targetだけが無効になるため、configureの終了値だけでなく対象の有効化理由とbuild結果を確認します。具体的な依存・CPU backendの選択は[ビルド案内](docs/BUILDING.ja.md)を参照してください。

<a id="benchmark-configuration-and-output"></a>

## サンプルが動いたら性能を測る

`examples/`には測定用CLIはありません。`benchmarks/`ではサイズ、warm-up、反復、試行、検証を制御できます。図の表示は**library**（データをGPUに保持した計算部）と**end-to-end**（準備・データ管理・結果取得を含む1回の処理）です。内部CLI／schema名の`compute`はlibraryに対応します。初回runtime起動やすべての解放処理を含むとは限らないため、[計測境界](docs/BENCHMARK_PROTOCOL.ja.md)を確認してください。

NVIDIAの[自分の環境での測定手順](docs/PORTABILITY.ja.md#measuring-on-your-own-system)、[Fortran測定の入力](nvidia/fortran/README.ja.md#first-checks-and-own-measurements)、AMDの[Sirius測定手順](jobs/sirius/README.md)を用途に応じて参照します。別GPUへ移すときは自分の環境に合わせ、サイト固有のパス・module・GPU targetを流用しません。

<a id="cpu-baselines-and-verification-notes"></a>

## 比較結果の読み方

CPU backendと実際の並列条件を確認します。要求スレッド数と実効worker数は同じとは限りません。C/C++ RAND／Thrustの評価は逐次CPU、Fortranでは逐次の組込み乱数・sumを使います。乱数はCPU／GPUで生成アルゴリズムが異なるため、同一アルゴリズムの高速化率ではありません。

AMD FFTは入力復元要件に合わせて計測しており、NVIDIA FFTとrepeat内の計時方式が異なります。[rocFFTの説明](amd/c-cpp/rocfft/README.ja.md#timing-scopes)を参照してください。

<a id="validation-status"></a>
<a id="publication-scope"></a>

## 公開物と確認範囲

NVIDIA C/C++・FortranとAMD C/C++のソース、設定、ツール、テスト、文書を公開しています。測定raw、実行ログ、PPTX、非配布の保存資料はcheckoutに含みません。ソースを利用するために、著者の保存データや会話履歴は必要ありません。

PegasusでNVIDIA、SiriusでAMDのbuild・数値検証・性能測定の記録があります。[検証報告](docs/VALIDATION_REPORT.md)と各分野の確認範囲を参照してください。これらは記録された版・環境の結果であり、別環境や後続変更の動作保証ではありません。CPU実効worker数や完全な実ロードライブラリ来歴など、未確認事項は保持しています。

管理主体は`ac2-prod`です。HAIRDESCの公式リポジトリではありません。

<a id="documentation-map"></a>
<a id="pegasus-execution-boundary"></a>
<a id="reference"></a>

## 詳しい情報

- [ビルドと実行](docs/BUILDING.ja.md)：最小の教材実行、依存、失敗時の確認。
- [測定手順](docs/PORTABILITY.ja.md)、[測定仕様](docs/BENCHMARK_PROTOCOL.ja.md)：自分の性能を測り、結果を解釈する。
- [問題・ソース仕様](docs/PROJECT_SPECIFICATION.md)、[結果schema](docs/RESULT_SCHEMA.md)：詳細な参照。
- [Pegasus](jobs/pegasus/README.md)、[Sirius](jobs/sirius/README.md)：そのシステムの利用者向け運用。
- [検証履歴](docs/VALIDATION_REPORT.md)、[開発計画](docs/IMPLEMENTATION_PLAN.md)、[決定記録](docs/DECISIONS.md)：保守・検証の経緯。

<a id="what-is-implemented"></a>

<details>
<summary>NVIDIA C/C++ソース一覧</summary>

| Library | CPU | CUDA | OpenACC | CPU benchmark | CUDA benchmark | OpenACC benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| cufft | [fft_cpu](nvidia/c-cpp/cufft/examples/fft_cpu.c) | [fft_gpu](nvidia/c-cpp/cufft/examples/fft_gpu.cu) | [openacc_cufft](nvidia/c-cpp/cufft/examples/openacc_cufft.cpp) | [fft_cpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_cpu_bench.c) | [fft_gpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_gpu_bench.cu) | [openacc_cufft_bench](nvidia/c-cpp/cufft/benchmarks/openacc_cufft_bench.cpp) |
| cublas | [blas_cpu](nvidia/c-cpp/cublas/examples/blas_cpu.c) | [blas_gpu](nvidia/c-cpp/cublas/examples/blas_gpu.cu) | [openacc_cublas](nvidia/c-cpp/cublas/examples/openacc_cublas.cpp) | [blas_cpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_cpu_bench.c) | [blas_gpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_gpu_bench.cu) | [openacc_cublas_bench](nvidia/c-cpp/cublas/benchmarks/openacc_cublas_bench.cpp) |
| cusparse | [sparse_cpu](nvidia/c-cpp/cusparse/examples/sparse_cpu.c) | [sparse_gpu](nvidia/c-cpp/cusparse/examples/sparse_gpu.cu) | [openacc_cusparse](nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp) | [sparse_cpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_cpu_bench.c) | [sparse_gpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_gpu_bench.cu) | [openacc_cusparse_bench](nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp) |
| cusolver | [solver_cpu](nvidia/c-cpp/cusolver/examples/solver_cpu.c) | [solver_gpu](nvidia/c-cpp/cusolver/examples/solver_gpu.cu) | [openacc_cusolver](nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp) | [solver_cpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_cpu_bench.c) | [solver_gpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_gpu_bench.cu) | [openacc_cusolver_bench](nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp) |
| curand | [rand_cpu](nvidia/c-cpp/curand/examples/rand_cpu.cpp) | [rand_gpu](nvidia/c-cpp/curand/examples/rand_gpu.cu) | [openacc_curand](nvidia/c-cpp/curand/examples/openacc_curand.cpp) | [rand_cpu_bench](nvidia/c-cpp/curand/benchmarks/rand_cpu_bench.cpp) | [rand_gpu_bench](nvidia/c-cpp/curand/benchmarks/rand_gpu_bench.cu) | [openacc_curand_bench](nvidia/c-cpp/curand/benchmarks/openacc_curand_bench.cpp) |
| thrust | [reduce_cpu](nvidia/c-cpp/thrust/examples/reduce_cpu.cpp) | [reduce_gpu](nvidia/c-cpp/thrust/examples/reduce_gpu.cu) | [openacc_thrust](nvidia/c-cpp/thrust/examples/openacc_thrust.cpp) | [reduce_cpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_cpu_bench.cpp) | [reduce_gpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_gpu_bench.cu) | [openacc_thrust_bench](nvidia/c-cpp/thrust/benchmarks/openacc_thrust_bench.cpp) |

</details>

<details>
<summary>既存測定・ツールの参照事項</summary>

Pegasusの既存CPU FFT系列は`cpu-fftw-threaded`です。既存の1ノードpilotと本評価の条件は[運用仕様](docs/PEGASUS_EXECUTION.md)を参照してください。`tools/run_suite.py`だけがノードraw-resultを書きます。既存の描画形式は実行時間の2パネル図で、speedup・throughput・reuse-count・amortized図は生成しません。Macのローカル／fixture検査では実CUDA GPUの動作確認を行わず、Pegasus／Siriusの実機記録とは区別しています。

</details>

<a id="license"></a>

## ライセンス

本リポジトリの独自作成部分には[MIT License（英語）](LICENSE)を適用します。

Copyright (c) 2026 Ryohei Kobayashi

既存の第三者の著作権表示と利用条件は引き続き適用されます。このライセンスは、外部ライブラリ本体や別途配布する教材スライド・動画のライセンスを変更するものではありません。
