# GPU Library Suite

[English](README.md) | [日本語](README.ja.md)

cuFFT、cuBLAS、cuSPARSE、cuSOLVER、cuRAND、Thrustを対象とした、移植可能なNVIDIA C/C++教材サンプル、再現可能なCPU/CUDA/OpenACCベンチマーク、厳密な結果処理ツール、Pegasus用ジョブ支援です。

実装済みのソース範囲は`nvidia/c-cpp`です。AMD、Fortran、その他のシステムへの対応は意図的に今後の対象とし、空の仮実装は置いていません。本プロジェクトは`ac2-prod`で管理しており、HAIRDESCの公式リポジトリではありません。

本リポジトリは**Library Edition (NVIDIA GPU, C/C++)**（ライブラリ編（NVIDIA GPU，C/C++））のソースリポジトリです。サンプルを読んだりビルドしたりするために、Pegasusへのアクセスや著者との会話履歴は必要ありません。GPUサンプルの実行には、対応するNVIDIA GPUと以下の依存環境が必要です。

<a id="publication-scope"></a>

## 公開範囲

今回の公開対象は、NVIDIA C/C++の教材・ベンチマークコード、既存の共通コード／ツール、設定、テスト、およびそれらの理解・ビルド・実行・測定に必要な文書です。汎用測定基盤の追加開発は、このコード公開の前提条件ではありません。

保存済み測定値、raw結果、実行ログ、メタデータ、作成済みのローカル配布アーカイブは、**今回の公開には含みません**。ソースとともに提供する教材測定データのダウンロード先はありません。[保存データからの再生成手順](docs/PORTABILITY.ja.md#regenerating-the-teaching-figures-from-saved-data)は、その入力を別途保有する読者だけを対象としています。以下の通常の利用手順では、読者自身の測定データを使います。

<a id="start-here"></a>

## はじめに

CPUのC/C++には慣れていてもCUDAは初めて、という場合は次の順で進めてください。

1. [cuBLASのCPUサンプル](nvidia/c-cpp/cublas/examples/blas_cpu.c)で、メモリ確保、全要素1の入力、DGEMM、結果確認の流れを読みます。その後、[直接CUDAを使う版](nvidia/c-cpp/cublas/examples/blas_gpu.cu)と[OpenACCでデータを管理する版](nvidia/c-cpp/cublas/examples/openacc_cublas.cpp)を比較してください。どちらのGPUプログラムもcuBLASを呼び出しますが、デバイスメモリとデータ移動の管理方法が異なります。
2. [cuBLAS：最初のビルドと実行](nvidia/c-cpp/cublas/README.ja.md#first-build-and-run)に従います。CPUサンプルから始め、利用可能なGPU版を実行して終了ステータスと数値結果を確認します。教材サンプルそのものは時間測定の実験ではありません。
3. [サンプルからベンチマークへの対応](nvidia/c-cpp/cublas/README.ja.md#from-example-to-benchmark)を読み、[自分のシステムで測定する手順](docs/PORTABILITY.ja.md#measuring-on-your-own-system)に沿って、小規模確認、設定、測定、検証、集計を行います。
4. [自分の測定データから図を作成](docs/PORTABILITY.ja.md#figures-from-your-own-measurements)します。図には観測した、または明示的に指定したマシン識別情報を使います。教材のマシンを前提にせず、その保存データも必要としません。
5. [図の読み方](docs/BENCHMARK_PROTOCOL.ja.md#reading-the-figures-and-applying-the-results)を参照し、compute/E2Eのコストを自分のアプリケーションと関連付けてください。他のライブラリのソース、依存環境、ビルド、検証の説明は、以下の表から参照できます。

| 読む場所 | 分かること |
| --- | --- |
| 各ライブラリの`examples/` | ベンチマーク基盤を伴わない、一連のライブラリ呼び出し全体 |
| 各ライブラリの`benchmarks/` | 実行時サイズ、スコープごとのリソース寿命、状態復元、時間測定、検証 |
| [`common/c-cpp`](common/c-cpp) | 共通の[CLI](common/c-cpp/src/cli.c)、[単調時計](common/c-cpp/src/clock.c)、[結果出力](common/c-cpp/src/result.c)、[GPU provenance](common/c-cpp/include/gpu_suite/cuda_metadata.hpp)。最初にここを読むのではなく、サンプルを読んだ後に参照してください |
| [`configs/pilot.json`](configs/pilot.json)、[`configs/benchmark.json`](configs/benchmark.json) | ワークロード、CPUバックエンド、検証閾値、スコープごとの反復回数と試行数 |
| [`tools/run_suite.py`](tools/run_suite.py) | 交互実行と、ノードraw結果の唯一の書込み主体 |
| [`tools/validate_results.py`](tools/validate_results.py)、[`tools/aggregate.py`](tools/aggregate.py)、[`tools/plot.py`](tools/plot.py) | rawレコードの確認、階層的な集計、2パネル図の生成 |
| [`jobs/pegasus`](jobs/pegasus) | サイト固有のモジュール、PBS、テレメトリ、回収処理。一般的なワークステーション利用の入口ではありません |

<a id="what-is-implemented"></a>

## 実装済みの内容

各ライブラリには、単一ソースで直接実行できる教材サンプル3本と、測定用ベンチマーク3本があります。CMakeターゲット名と実行ファイル名は、ソースファイル名から拡張子を除いた名前と一致します。

| ライブラリ | CPU | CUDA | OpenACC | CPUベンチマーク | CUDAベンチマーク | OpenACCベンチマーク |
| --- | --- | --- | --- | --- | --- | --- |
| [cuFFT](nvidia/c-cpp/cufft/README.ja.md) | [fft_cpu](nvidia/c-cpp/cufft/examples/fft_cpu.c) | [fft_gpu](nvidia/c-cpp/cufft/examples/fft_gpu.cu) | [openacc_cufft](nvidia/c-cpp/cufft/examples/openacc_cufft.cpp) | [fft_cpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_cpu_bench.c) | [fft_gpu_bench](nvidia/c-cpp/cufft/benchmarks/fft_gpu_bench.cu) | [openacc_cufft_bench](nvidia/c-cpp/cufft/benchmarks/openacc_cufft_bench.cpp) |
| [cuBLAS](nvidia/c-cpp/cublas/README.ja.md) | [blas_cpu](nvidia/c-cpp/cublas/examples/blas_cpu.c) | [blas_gpu](nvidia/c-cpp/cublas/examples/blas_gpu.cu) | [openacc_cublas](nvidia/c-cpp/cublas/examples/openacc_cublas.cpp) | [blas_cpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_cpu_bench.c) | [blas_gpu_bench](nvidia/c-cpp/cublas/benchmarks/blas_gpu_bench.cu) | [openacc_cublas_bench](nvidia/c-cpp/cublas/benchmarks/openacc_cublas_bench.cpp) |
| [cuSPARSE](nvidia/c-cpp/cusparse/README.ja.md) | [sparse_cpu](nvidia/c-cpp/cusparse/examples/sparse_cpu.c) | [sparse_gpu](nvidia/c-cpp/cusparse/examples/sparse_gpu.cu) | [openacc_cusparse](nvidia/c-cpp/cusparse/examples/openacc_cusparse.cpp) | [sparse_cpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_cpu_bench.c) | [sparse_gpu_bench](nvidia/c-cpp/cusparse/benchmarks/sparse_gpu_bench.cu) | [openacc_cusparse_bench](nvidia/c-cpp/cusparse/benchmarks/openacc_cusparse_bench.cpp) |
| [cuSOLVER](nvidia/c-cpp/cusolver/README.ja.md) | [solver_cpu](nvidia/c-cpp/cusolver/examples/solver_cpu.c) | [solver_gpu](nvidia/c-cpp/cusolver/examples/solver_gpu.cu) | [openacc_cusolver](nvidia/c-cpp/cusolver/examples/openacc_cusolver.cpp) | [solver_cpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_cpu_bench.c) | [solver_gpu_bench](nvidia/c-cpp/cusolver/benchmarks/solver_gpu_bench.cu) | [openacc_cusolver_bench](nvidia/c-cpp/cusolver/benchmarks/openacc_cusolver_bench.cpp) |
| [cuRAND](nvidia/c-cpp/curand/README.ja.md) | [rand_cpu](nvidia/c-cpp/curand/examples/rand_cpu.cpp) | [rand_gpu](nvidia/c-cpp/curand/examples/rand_gpu.cu) | [openacc_curand](nvidia/c-cpp/curand/examples/openacc_curand.cpp) | [rand_cpu_bench](nvidia/c-cpp/curand/benchmarks/rand_cpu_bench.cpp) | [rand_gpu_bench](nvidia/c-cpp/curand/benchmarks/rand_gpu_bench.cu) | [openacc_curand_bench](nvidia/c-cpp/curand/benchmarks/openacc_curand_bench.cpp) |
| [Thrust](nvidia/c-cpp/thrust/README.ja.md) | [reduce_cpu](nvidia/c-cpp/thrust/examples/reduce_cpu.cpp) | [reduce_gpu](nvidia/c-cpp/thrust/examples/reduce_gpu.cu) | [openacc_thrust](nvidia/c-cpp/thrust/examples/openacc_thrust.cpp) | [reduce_cpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_cpu_bench.cpp) | [reduce_gpu_bench](nvidia/c-cpp/thrust/benchmarks/reduce_gpu_bench.cu) | [openacc_thrust_bench](nvidia/c-cpp/thrust/benchmarks/openacc_thrust_bench.cpp) |

`examples/`は教材掲載コードの正本です。これらのプログラムはベンチマークCLIや機械可読出力を持ちません。`benchmarks/`では、2種類の測定スコープ、warm-up、repeat/試行制御、検証、UTC時刻、厳密なJSONLまたはCSVレコードを追加します。問題とファイル名の厳密な定義は[`docs/PROJECT_SPECIFICATION.md`（英語）](docs/PROJECT_SPECIFICATION.md)、測定の意味は[`docs/BENCHMARK_PROTOCOL.ja.md`](docs/BENCHMARK_PROTOCOL.ja.md)を参照してください。

<a id="requirements-and-cpu-only-build"></a>

## 必要環境とCPU-onlyビルド

CMake 3.20以降、C17コンパイラ、C++17コンパイラが必要です。PythonツールはPython 3.9以降に対応します。CUDA、NVHPC、FFTW、oneMKL、OpenBLAS、LAPACKEは任意依存です。CMakeのコンパイル・リンク検査が、影響するターゲットだけを無効化し、その理由を表示します。依存パッケージを自動インストールすることはありません。

ローカルのCPU-onlyビルドでは、CUDA言語の検査や有効化は行いません。

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-only-validation \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON \
  -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF
cmake --build /tmp/gpu-library-suite-local-build/cpu-only-validation --parallel
ctest --test-dir /tmp/gpu-library-suite-local-build/cpu-only-validation \
  --output-on-failure
```

このプロファイルでは、外部依存を持たない`rand_cpu`、`rand_cpu_bench`、`reduce_cpu`、`reduce_cpu_bench`が常に利用可能です。教材サンプルの直接コンパイルと、任意プロバイダの必要条件は各ライブラリのREADMEに記載しています。

production成果物には、重複しない2つのビルドプロファイルを使います。

- `cpu-cuda`：CPU ON、CUDA ON、OpenACC OFF。
- `openacc`：`nvc`/`nvc++`、CPU OFF、CUDA OFF、OpenACC ON。

CUDAアーキテクチャとNVHPC GPUターゲットは外部から与えるビルド入力です。OpenACCのビルドツリーでは、CMakeが選択するCUDA ToolkitとNVHPCが選択するToolkitが一致する必要があります。[`docs/PORTABILITY.ja.md`](docs/PORTABILITY.ja.md)と[PegasusジョブのREADME（英語）](jobs/pegasus/README.md)を参照してください。

CMakeのビルドメタデータには、コンパイラ言語ごとの`global_configure_flags`を記録します。このフィールドは意図的にグローバルなCMake configureフラグだけを対象とし、ターゲットのコンパイル定義/オプション、プロバイダフラグ、OpenACC相互運用オプション、リンクオプションを表すものではありません。OpenACCとThrustの相互運用フラグは、ビルドメタデータに別途記録します。

`.git`のないソースコピーでも、`git_metadata_available=false`、`git_commit=null`、`git_dirty=null`を持つ有効なメタデータを生成します。アーカイブ／ローカル検証ではこのソースをビルドできますが、production実行ではGit provenanceが不明な状態をcleanとは扱わず拒否します。

<a id="benchmark-configuration-and-output"></a>

## ベンチマークの設定と出力

[`configs/pilot.json`](configs/pilot.json)と[`configs/benchmark.json`](configs/benchmark.json)は、承認済みの3サイズの教材掲載用ワークロードとcompute-repeat候補を共通に保持します。pilotは1試行、productionは5試行です。どちらも`compute`と`end-to-end`を独立に設定し、end-to-endのrepeatは1です。Pegasusで1ノードpilotを行った後、OOM、検証、walltime、短すぎる測定区間の問題が実証された場合には、人間がこれらの正本ファイルを一度改訂できます。日付や版番号を付けた代替ファイルは作成しません。

個別のベンチマークでは、次の出力規則を使います。

- `--output -`：機械可読レコードだけを標準出力へ、診断を標準エラー出力へ出力します。
- `--output <path>`：新規ファイルを排他的に作成し、上書きや暗黙の追記を拒否します。

スイート実行では、[`tools/run_suite.py`](tools/run_suite.py)がすべてのベンチマークを`--output -`で起動し、レコードを検証して、必要に応じて失敗/skipped行を生成します。ノード単位のraw-resultファイルを書き込むのは、このランナーだけです。`--dry-run`は、解決済みの決定的なJSONを表示し、ファイル、ディレクトリ、メタデータ、プロセスは作成しません。

通常のデータの流れは次のとおりです。

```text
partial manifests -> merge_manifests.py -> run_suite.py -> validate_results.py
                                             |
                                             +-> aggregate.py -> plot.py
```

出力先はすべて新規の排他的作成です。raw行は、開始後に失敗した試行（`attempted=true`、`status=failure`）と、開始されなかった残りの試行（`attempted=false`、`status=skipped`）を区別します。スキーマと厳密なprovenanceの契約は[`docs/RESULT_SCHEMA.md`（英語）](docs/RESULT_SCHEMA.md)に定義されています。

productionスイートの各呼び出しでは、実効設定に含まれる完全な検証オブジェクトを渡します。組込みCLIの既定値は、単独の教材／smoke用途だけのものであり、production閾値の別の定義元にはなりません。

実行時の次元、ライブラリAPIの整数引数、要素数の積、確保バイト数は、メモリ確保またはライブラリ呼び出しの前に確認します。対応範囲外は構造化されたprerequisite／ベンチマーク結果として報告し、小さいワークロードへ切り詰めることはありません。warm-upは選択したスコープに従います。computeは永続コンテキストを再利用し、end-to-endは完全な一時パイプラインを使い、warm-upのリソースを残しません。入力・中間値・指標のNaNや無限大は、明示的な`verification_status=nonfinite`の失敗として扱います。主要指標をJSONの`null`として保存し、非標準の`NaN`/`Infinity` tokenは出力しません。厳密な規則は[`docs/BENCHMARK_PROTOCOL.ja.md`](docs/BENCHMARK_PROTOCOL.ja.md)と[`docs/RESULT_SCHEMA.md`（英語）](docs/RESULT_SCHEMA.md)を参照してください。

教材掲載用出力は、ライブラリごとの実行時間図6枚です。各図はデータ常駐のcomputeパネルと、ホスト入力からホスト出力までを1回で処理するパネルで構成し、単位はミリ秒、CPU・CUDA・OpenACCを同じ図に示します。compute repeatは測定区間を増幅するだけで、プロット値は常に1演算あたりの経過時間です。speedup、throughput、reuse-count、amortized、その他の汎用図は生成しません。描画の性能値はwave間要約レコードだけから読み取ります。raw結果は、Thrust図を出力する前にCUDA/OpenACCの`library_version`が一致する根拠を要求するためだけに使用します。図とキャプションの厳密な規則は[`docs/BENCHMARK_PROTOCOL.ja.md`](docs/BENCHMARK_PROTOCOL.ja.md)にあります。`tools/plot.py`の位置引数にaggregate JSONLを渡し、必須の`--raw-results`にはキャンペーンのノードraw JSONLファイルを渡します。

<a id="cpu-baselines-and-verification-notes"></a>

## CPU基準実装と検証上の注意

Pegasusの正本教材掲載用設定では、cuFFTのCPU系列は`cpu-fftw-threaded`だけです。逐次FFTW実行ファイルは教材との対応確認用として別途利用できますが、教材掲載用系列には含まず、threaded FFTWの代わりにはしません。

cuRANDの`cpu-std-random-serial`とThrustの`cpu-stl-serial`はproductionの逐次実装です。CPUスレッド数として48を要求していても、承認済み教材掲載用凡例では**single thread**と表示します。並列または同一アルゴリズムの基準実装とは呼ばず、教材掲載用描画ではこれらを使ったspeedupを計算しません。cuRANDは同じ分布・出力型のタスクをCPU/GPU間で比較しますが、RNGアルゴリズムは異なるため、要素ごとの一致は要求しません。

<a id="pegasus-execution-boundary"></a>

## Pegasus実行の責任範囲

[`jobs/pegasus`](jobs/pegasus)以下のファイルは、ビルド・ジョブのrender、ハッシュ付き実行環境の取得、キャンペーン/waveメタデータの準備、ノード障害の分離、テレメトリ区間の対応付け、ノード成果物の回収を行います。スケジューラへの投入、照会、取消しなどの操作は行いません。アカウント、キュー、モジュールバージョン、Toolkit選択、アーキテクチャ、出力パスは人間が指定します。

各ノードは、自身のGPU名、UUID、NVIDIAドライバパッケージの識別情報、およびCUDA Driver API/Runtimeバージョンを、ノード上の`libcudart`から取得します。GPUベンチマークはCUDAデバイス/ランタイムとライブラリAPIを独立に照会し、成功行はそのノードメタデータと一致する必要があります。1ノードの識別情報を全ノードへ使い回しません。ビルド済みバイナリでは、`nvcc`、`nvc`、`nvc++`は任意のprovenance検査ですが、ドライバ、ランタイム、解決済み共有ライブラリは実行時の必須条件です。

回収できたノードはベンチマークまたは検証の失敗を記録しますが、他ノードの成果物回収を継続できるよう通常は0で終了します。測定起動後、ジョブのマスターのコレクタが予定した全ノードを確認し、ジョブを0以外の終了値で終了させるかどうかを最終決定します。正本手順は[`docs/PEGASUS_EXECUTION.md`（英語）](docs/PEGASUS_EXECUTION.md)、人間向けチェックリストは[`docs/PEGASUS_MANUAL_VALIDATION.md`（英語）](docs/PEGASUS_MANUAL_VALIDATION.md)です。

<a id="validation-status"></a>

## 検証状況

ローカル検証は、厳密なシリアライズ、スキーマ、全ソース一覧、依存関係検査フィクスチャ、CPUベンチマークフィクスチャ、ランナー／集計／描画ロジック、Python 3.9互換性、シェル構文、利用可能な場合のShellCheck、ジョブのレンダリング、テレメトリ解析、複数ノード障害の模擬実行を対象としています。[`docs/VALIDATION_REPORT.md`（英語）](docs/VALIDATION_REPORT.md)を参照してください。

模擬プロバイダフィクスチャと`.git`なしソースコピーのテストを含む、cleanなLinux GCC/Clang CPU-onlyビルドはmerge前の必須条件です。最小ワークフローは[`.github/workflows/cpu-linux.yml`](.github/workflows/cpu-linux.yml)にあります。ワークフローの定義そのものは、いずれかのjobが実行された証拠ではありません。

このローカル検証では、実CUDA GPU、CUDA Toolkit、NVHPCコンパイラ、Pegasusスケジューラ、production用CPUライブラリインストール環境は使用していません。構文のみのテストや模擬プロバイダテストから、それらの確認済み状態を推論してはいけません。別途保存されたPegasus記録には実際のビルド、サンプル、memcheck、測定結果があり、承認済み描画commitではLinux CIも実行されています。コード版、範囲、残る制限は[保存済み実行根拠（英語）](docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)を参照してください。最初の読者ガイド確認は静的確認で、その後の[読者向けツールテストと非公開データのoffline再生成（英語）](docs/VALIDATION_REPORT.md#reader-tools-and-offline-replay-validation)は別途完了しています。いずれも、任意の環境で読者向けワークフロー全体のGPU実行が確認できたことを意味しません。非公開の再生成入力は今回のコード公開に含みません。

<a id="documentation-map"></a>

## 文書案内

- [`AGENTS.md`（英語）](AGENTS.md)：恒常的なリポジトリ規則と正本の参照先。
- [`docs/PROJECT_SPECIFICATION.md`（英語）](docs/PROJECT_SPECIFICATION.md)：階層、正本ソース、教材の問題、OpenACCデータ規則。
- [`docs/BENCHMARK_PROTOCOL.ja.md`](docs/BENCHMARK_PROTOCOL.ja.md)：CLI、時間測定、検証、順序、統計。
- [`docs/RESULT_SCHEMA.md`（英語）](docs/RESULT_SCHEMA.md)：設定、raw／aggregateレコード、マニフェスト、メタデータ、ハッシュ。
- [`docs/PEGASUS_EXECUTION.md`（英語）](docs/PEGASUS_EXECUTION.md)：Pegasusのビルド/run、provenance、テレメトリ、回収動作。
- [`docs/IMPLEMENTATION_PLAN.md`（英語）](docs/IMPLEMENTATION_PLAN.md)：段階ごとの条件と受入基準。
- [`docs/DECISIONS.md`（英語）](docs/DECISIONS.md)：採用した判断と理由。
- [`docs/PORTABILITY.ja.md`](docs/PORTABILITY.ja.md)：再利用と依存環境の境界。
- [`docs/ADDING_A_NEW_VENDOR.md`（英語）](docs/ADDING_A_NEW_VENDOR.md)、[`docs/ADDING_FORTRAN.md`（英語）](docs/ADDING_FORTRAN.md)：将来の拡張ガイド。

<a id="license"></a>

## ライセンス

本リポジトリの独自作成部分には[MIT License（英語）](LICENSE)を適用します。

Copyright (c) 2026 Ryohei Kobayashi

既存の第三者の著作権表示と利用条件は引き続き適用されます。このライセンスは、外部ライブラリ本体や別途配布する教材スライド・動画のライセンスを変更するものではありません。
