# cuBLAS C/C++のサンプルとベンチマーク

[English](README.md) | [日本語](README.ja.md)

このディレクトリでは、列優先FP64 DGEMMの正本問題を、CBLAS、直接cuBLAS、OpenACC/cuBLASで比較します。問題の数学的定義は[`docs/PROJECT_SPECIFICATION.md`（英語）](../../../docs/PROJECT_SPECIFICATION.md)、CLI、時間測定、状態復元、検証は[`docs/BENCHMARK_PROTOCOL.ja.md`](../../../docs/BENCHMARK_PROTOCOL.ja.md)が定めます。

<a id="canonical-sources-and-targets"></a>

## 正本ソースとターゲット

| 用途 | ソース | CMakeターゲット／実行ファイル |
| --- | --- | --- |
| CPU教材 | `examples/blas_cpu.c` | `blas_cpu` |
| CUDA教材 | `examples/blas_gpu.cu` | `blas_gpu` |
| OpenACC教材 | `examples/openacc_cublas.cpp` | `openacc_cublas` |
| CPUベンチマーク | `benchmarks/blas_cpu_bench.c` | `blas_cpu_bench` |
| CUDAベンチマーク | `benchmarks/blas_gpu_bench.cu` | `blas_gpu_bench` |
| OpenACCベンチマーク | `benchmarks/openacc_cublas_bench.cpp` | `openacc_cublas_bench` |

<a id="teaching-default"></a>

## 教材の既定問題

3本の教材プログラムはすべて、列優先FP64行列を使い、`alpha=1`、`beta=1`で1,024×1,024のDGEMMを計算します。ベンチマーク基盤を含まない、直接実行できる完全な単一ソースプログラムです。

<a id="dependencies"></a>

## 依存環境

CPUターゲットには、`cblas_dgemm`をコンパイル・リンクできるONEMKL、OPENBLAS、GENERIC_CBLASのいずれかを明示的に選択する必要があります。`BLAS_FOUND`だけでは十分としません。プロバイダごとのキャッシュキーと所定のヘッダ/ライブラリ名により、oneMKLヘッダとOpenBLASライブラリの混在を防ぎます。CUDAターゲットにはCUDA RuntimeとcuBLAS、OpenACCターゲットにはさらにNVHPC OpenACC C++が必要です。

<a id="direct-compile"></a>

## 直接コンパイル

CPUソースは、CMakeがプロバイダ指定を上書きしない限り、既定で`<cblas.h>`をincludeします。次の代表的なコマンドは、通常の検索パスが設定済みであることを前提とします。

```bash
mkdir -p /tmp/gpu-library-suite-local-build
cc -std=c17 nvidia/c-cpp/cublas/examples/blas_cpu.c \
  -lcblas -lm -o /tmp/gpu-library-suite-local-build/blas_cpu-direct
nvcc -std=c++17 nvidia/c-cpp/cublas/examples/blas_gpu.cu \
  -lcublas -o /tmp/gpu-library-suite-local-build/blas_gpu-direct
nvc++ -std=c++17 -acc -cuda \
  nvidia/c-cpp/cublas/examples/openacc_cublas.cpp -cudalib=cublas \
  -o /tmp/gpu-library-suite-local-build/openacc_cublas-direct
```

<a id="first-build-and-run"></a>

## 最初のビルドと実行

リポジトリのルートを作業ディレクトリとするシェルを使ってください。CMake 3.20以降、C17/C++17ツールチェーン、コンパイラ／リンカから利用できるCBLAS実装が必要です。この最初のCPU実行にCUDAのインストールやPegasusアカウントは不要です。以下は`GENERIC_CBLAS`（`cblas.h`と`libcblas`）を選択します。oneMKLまたはOpenBLASを使う場合は、`ONEMKL`または`OPENBLAS`を明示的に選択してください。変更を記録せずプロバイダを置き換えないでください。oneMKLの検出はインストール済みの`MKLROOT`を使います。実際のプロバイダインストール環境を、通常のインクルード／ライブラリ検索パスまたはCMakeキャッシュエントリで示す必要があります。

```bash
EXAMPLE_BUILD="$(mktemp -d "${TMPDIR:-/tmp}/gpu-library-suite-example.XXXXXX")"
cmake -S . -B "$EXAMPLE_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=OFF -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_CPU_BLAS_BACKEND=GENERIC_CBLAS
cmake --build "$EXAMPLE_BUILD" --target blas_cpu
"$EXAMPLE_BUILD/nvidia/c-cpp/cublas/blas_cpu"
printf 'example exit status: %s\n' "$?"
```

configureで`blas_cpu`が無効化された場合やビルドが失敗した場合は停止してください。任意依存の欠如をCPU/GPU比較の成功とは扱いません。成功時は`CBLAS DGEMM complete; max error = ...`と表示し、0で終了します。入力は全要素1なので、出力の全要素は`1025`（`k + 1`）になります。許容する最大絶対誤差は`1e-10`以下です。終了値が0以外、または誤差がこれを超える場合は調査が必要です。

続いて、対応するNVIDIA GPU/ドライバ、CUDA Toolkit、およびOpenACC用のNVHPCを用意します。次節のToolkit/アーキテクチャ変数を明示的に設定し、2つのプロファイルをビルドします。コマンドの既定ディレクトリは以下のとおりです。一般的な利用手順に沿って進めた場合は、そこでexportしたディレクトリの値を維持してください。対応ターゲットのビルド成功後にだけ実行します。

```bash
CPU_CUDA_BUILD="${CPU_CUDA_BUILD:-/tmp/gpu-library-suite-local-build/cpu-cuda}"
OPENACC_BUILD="${OPENACC_BUILD:-/tmp/gpu-library-suite-local-build/openacc}"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_cpu"
printf 'CPU exit status: %s\n' "$?"
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_gpu"
printf 'CUDA exit status: %s\n' "$?"
"$OPENACC_BUILD/nvidia/c-cpp/cublas/openacc_cublas"
printf 'OpenACC exit status: %s\n' "$?"
```

最後のコマンドだけでなく、各コマンドの終了ステータスを直後に確認してください。GPUプログラムでも同じ`max error`の条件を確認し、メッセージには`cuBLAS`または`OpenACC-managed cuBLAS`を表示します。ベンチマークの時間は表示しません。GPUの依存環境がない場合でもCPUサンプルは有用ですが、3実装の実行確認は未完了のままです。

<a id="cmake-configure-and-build"></a>

## CMakeのconfigureとビルド

以下はgeneric CBLASを使うCPU/CUDAツリーと、別のOpenACCツリーの例です。`CUDA_TOOLKIT_ROOT`、`CUDA_ARCHITECTURES`、`NVHPC_GPU_TARGET`に、サイトで承認された具体値が設定済みであることを前提とします。

```bash
cmake -S . -B /tmp/gpu-library-suite-local-build/cpu-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DGPU_SUITE_CPU_BLAS_BACKEND=GENERIC_CBLAS
cmake --build /tmp/gpu-library-suite-local-build/cpu-cuda \
  --target blas_cpu blas_gpu blas_cpu_bench blas_gpu_bench

NVHPC_CUDA_HOME="${CUDA_TOOLKIT_ROOT}" \
cmake -S . -B /tmp/gpu-library-suite-local-build/openacc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DCUDAToolkit_ROOT="${CUDA_TOOLKIT_ROOT}" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="${NVHPC_GPU_TARGET}" \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON
cmake --build /tmp/gpu-library-suite-local-build/openacc \
  --target openacc_cublas openacc_cublas_bench
```

<a id="benchmark-cli"></a>

## ベンチマークCLI

```bash
/tmp/gpu-library-suite-local-build/cpu-cuda/nvidia/c-cpp/cublas/blas_cpu_bench \
  --size 128 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --output - --format jsonl \
  --cpu-backend cpu-generic-cblas --cpu-threads 48 \
  --cpu-backend-role production --series-role primary \
  --cpu-parallelism threaded
```

選択したプロバイダに対応するCPUバックエンド名を使ってください。有効スレッド数が独立に確認できている場合は、`--cpu-threads-effective`で渡します。それ以外はnullのままにします。productionスイート実行の閾値・roleは、すべて実効設定から取得する必要があります。

<a id="timing-scopes"></a>

## 測定スコープ

`compute`は、メモリ／ハンドルの作成とCの復元を測定開始前に行い、repeatを通してDGEMMの更新を連続適用します。そのため検証では、repeat回の更新を反映した結果を確認します。`end-to-end`は各repeatのメモリ確保、コピー、ハンドルのライフサイクル、DGEMM、完了、結果取得を測定し、後処理は終了時刻の後に行います。各repeatの前に正本のCを復元するため、状態を引き継ぎません。

<a id="from-example-to-benchmark"></a>

## サンプルからベンチマークへの対応

対応する`_bench`ソースより先に、3本の短いサンプルを読んでください。どちらも全要素1の同じDGEMM問題を解きますが、ベンチマークでは`main`関数全体の時間を測るのではなく、測定条件を制御します。

| プログラムの部分 | 教材サンプル | ベンチマークでの対応 |
| --- | --- | --- |
| 入力と初期化 | `m=n=k=1024`に固定し、ホストのA/B/Cを1で埋める | CLI／設定で次元を指定し、サイズを確認。同じ正本入力を測定区間外で生成 |
| CPU計算 | `cblas_dgemm`を1回呼ぶ | 同じプロバイダ呼び出し。並列化はCPUライブラリ内部で行い、アプリケーションのOpenMPループではない |
| 直接CUDAのメモリ | 明示的な`cudaMalloc`、H2D、D2H | computeでは永続的なメモリ確保/ハンドル、E2Eではrepeatごとのパイプライン |
| OpenACCのメモリ | `acc data`が配列を所有し、`host_data use_device`でデバイスポインタをcuBLASへ渡す | 所有関係は同じ。スコープに応じてデータ領域への出入りとタイムスタンプの位置関係を変える |
| 計算 | `cublasDgemm`または`cblas_dgemm`を1回呼ぶ | computeでは反復呼び出し、教材掲載用設定のE2Eでは1回の処理 |
| 復元と検証 | 1回呼び出した期待値`k+1`を確認 | warm-up後と各試行前にCを復元。computeの検証は全repeatの更新を反映し、E2Eでは各repeat前にCを復元 |
| 時間測定と出力 | タイマーや機械可読レコードはない | 共通の単調wall clock、GPU完了の同期、試行ごとのJSONL/CSV、測定区間外の検証 |

正本の[時間測定境界](../../../docs/BENCHMARK_PROTOCOL.ja.md#timing-fields)には、どの準備、転送、同期、後処理が測定に含まれるかを厳密に記載しています。教材プログラムを単にストップウォッチで囲むのではなく、[小規模測定とデータ処理全体の手順](../../../docs/PORTABILITY.ja.md#measuring-on-your-own-system)へ進んでください。

<a id="verification"></a>

## 検証

実効設定を使い、最大絶対誤差を`abs_tolerance + rel_tolerance * reference_scale`と比較します。検証失敗はraw失敗として保持し、状態を復元した後の試行は継続できます。

<a id="cpu-backend-and-role"></a>

## CPUバックエンドとrole

設定したCBLASプロバイダが、productionのprimary CPUバックエンドです。対応する名前は`cpu-onemkl`、`cpu-openblas`、`cpu-generic-cblas`です。キャンペーンには1つを選択し、他プロバイダへ暗黙にフォールバックすることはありません。

<a id="openacc-notes"></a>

## OpenACCの注意点

OpenACCデータ領域がA、B、Cを所有します。`host_data use_device`でそのアドレスをcuBLASへ渡し、別のCUDAメモリ確保が配列を重複所有することはありません。正本のリンク対象は`OpenACC::OpenACC_CXX`、`CUDA::cudart`、`CUDA::cublas`です。

<a id="known-limitations-and-local-validation"></a>

## 既知の制限とローカル検証

ランタイムから信頼できる値を取得できない場合、プロバイダのスレッド数は不明のままです。そのため、要求スレッド数を有効スレッド数と読み替えません。ローカルテストはfake CBLASプロバイダとGPU構文だけを対象としています。実cuBLASランタイム、NVHPCコンパイラ、GPU、production用oneMKL/OpenBLASインストール環境、Pegasus実行はローカルでは未検証です。その後に保存された実機実行記録は、[検証報告（英語）](../../../docs/VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)でこれらのローカルテストと区別しています。
