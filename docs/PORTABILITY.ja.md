# 移植性ガイド

[English](PORTABILITY.md) | [日本語](PORTABILITY.ja.md)

<a id="purpose"></a>

## 目的

このガイドは、サイト固有の前提を移植可能なソースへ持ち込まずに、実装済みNVIDIA C/C++スイートを再利用する方法をまとめます。別のビルド仕様やベンチマーク仕様ではありません。恒常的な方針は[`AGENTS.md`（英語）](../AGENTS.md)、ビルド要件は[`IMPLEMENTATION_PLAN.md`（英語）](IMPLEMENTATION_PLAN.md)、時間測定とスキーマの動作は[`BENCHMARK_PROTOCOL.ja.md`](BENCHMARK_PROTOCOL.ja.md)と[`RESULT_SCHEMA.md`（英語）](RESULT_SCHEMA.md)が定めます。

以下の既定build・コマンド例はC/C++用です。[Fortran案内](../nvidia/fortran/README.ja.md)では専用コンパイラ/providerのbuildと診断configを示し、その後のruntime取得・検証・集計・図生成を本ガイドと共用します。言語別にbuild tree・manifest・runを分離してください。FortranのPegasusビルド・GPU確認・本測定は[検証報告（英語）](VALIDATION_REPORT.md#fortran-production-validation)に記録しています。任意のコンパイラ・GPU・非PBS環境の動作保証ではありません。

<a id="portable-layers"></a>

## 移植可能な層

- `nvidia/c-cpp/*/examples`は、サイトパス、スケジューラオプション、アーキテクチャフラグを含まない直接的な教材プログラムです。
- `nvidia/c-cpp/*/benchmarks`は、ワークロードと実行制御をすべて共通CLIから受け取る、単一プロセスのベンチマークプログラムです。
- `common/c-cpp`は、C++17からも使える厳密なC17の測定、検査付き算術、CLI、シリアライズ、メタデータ処理を提供します。
- `configs`、`tools`、`tests`の設定・実行・検証・集計コードは、Python 3.9以降の標準ライブラリの動作を使います。PNG生成にはmatplotlibも必要です。図のラベルには記録された識別情報または明示的な表示設定を使い、不明なハードウェアを教材のマシンに置き換えません。cuSOLVERの目盛りは実際の問題サイズに従います。
- `jobs/<system>`だけが、スケジューラ、モジュールシステム、ノードローカルファイルシステム、サイトの起動規約を扱う層です。

<a id="build-boundaries"></a>

## ビルドの境界

最上位のCMakeプロジェクトは、最初のconfigureではCとC++だけを有効にします。直接CUDAは要求された場合にだけ確認・有効化します。FFTW、CBLAS、oneMKL Sparse、LAPACKE、CUDA Toolkit、OpenACC、Thrust連携が欠けると、影響するターゲットを無効化し、configureの結果一覧に理由を示します。ヘッダやライブラリ名を検出しただけでは十分としません。

productionではビルドツリーを分離します。

| プロファイル | 内容 |
| --- | --- |
| `cpu-cuda` | CPUと直接CUDAのターゲット。OpenACCターゲットは含まない |
| `openacc` | NVHPC OpenACCのターゲット。CPUや直接CUDAのターゲットは含まない |

2つのプロバイダビルドを曖昧なマニフェスト識別情報のもとでmergeしないでください。各成果物にはターゲット、ビルドプロファイル、バックエンド種別、コンパイラ/ビルドメタデータ、バイナリハッシュを記録します。CUDAアーキテクチャとNVHPC GPUターゲットは外部入力であり、実行時にデバイスから推測しません。

<a id="numerical-portability"></a>

## 数値面の移植性

プロジェクト方針では、未承認のfast math、TF32、Tensor Coreモード、ハードウェア固有の数値変更を禁止しています。ただし、この方針は、継承されたコンパイラ既定値が数値的に厳密である証拠ではありません。保存済みNVHPC OpenACC Releaseメタデータには`-fast -O3 -DNDEBUG`が含まれ、現在のCMake層は`-fast`を記録しますが拒否しません。ターゲットごとのフラグと、文書化された浮動小数点上の影響は[保存済みビルドの監査（英語）](VALIDATION_REPORT.md#nvhpc-release-flag-audit)で確認していますが、方針上の承認と当時の実行時FPU状態は未解決です。特に、後方の`-O3`は`-fast`の他の構成要素を取り消しません。保存条件を黙って変更したり、厳密な数値演算として監査済みと説明したりしないでください。

CPU、CUDA、OpenACCの各系列は、同じ設定の問題、精度、スコープ、repeat数、検証の契約を使います。メモリ不足は明示的な試行の失敗であり、問題を黙って小さくすることはありません。

production CPUプロバイダは明示的に指定します。referenceまたはauxiliary実装で、欠けたproductionの比較基準を置き換えることはできません。特に、primary cuFFT speedupでthreaded FFTWを逐次FFTWに置き換えません。

<a id="runtime-portability"></a>

## 実行環境の移植性

キャンペーンの実効設定、ソース状態、実行ファイルマニフェスト、バイナリハッシュ、ビルドメタデータ、実行環境ハッシュは結果とともに保存します。ランタイムハッシュが異なる結果には新しいrun IDが必要であり、wave間で暗黙にmergeしません。

逐次バックエンドが有効スレッド数1を使う場合も、8個のCPUランタイム変数を保持します。Linuxではランタイム収集時に`ldd`を記録します。利用できないローカルコマンドはunavailableと報告し、推測した出力で置き換えません。サイト固有コレクタはテレメトリを拡張できますが、必須rawレコードやUTC時刻との対応付けを省略できません。

<a id="porting-checklist"></a>

## 移植前の確認項目

別のシステムでスイートを使う前に、次を確認してください。

1. CPU-onlyツリーをconfigureし、すべての有効化／無効化理由を確認する。
2. 外部からアーキテクチャとToolkitの値を選択し、CUDAとOpenACCを別ツリーでビルドする。
3. マニフェストをmergeし、プロバイダ/プロファイルの衝突をすべて拒否する。
4. 調整前に、教材サンプルと、小規模で検証付きのpilotを実行する。
5. 完全な実行環境とバイナリ依存関係を取得する。
6. 集計前にraw結果を検証する。
7. 未実行の任意バックエンドを成功と報告せず、それぞれ文書に明示する。

新しいvendorや言語の追加はソース範囲の変更であり、ビルドフラグを増やすだけではありません。プロジェクト所有者が拡張を承認した後にだけ、[`ADDING_A_NEW_VENDOR.md`（英語）](ADDING_A_NEW_VENDOR.md)または[`ADDING_FORTRAN.md`（英語）](ADDING_FORTRAN.md)に従ってください。

<a id="measuring-on-your-own-system"></a>

## 自分のシステムで測定する

これは読者向けの利用手順であり、[ベンチマークプロトコル](BENCHMARK_PROTOCOL.ja.md)や[結果スキーマ（英語）](RESULT_SCHEMA.md)の代わりではありません。以下のシェルブロックはすべて、**リポジトリのルートから、同じBashセッションで**実行します。コマンドはソースとCLIに照らして確認した手順であり、読者のマシンでこのワークフローを実行済みという主張ではありません。生成ファイルは各回の新しいrunディレクトリに保持してください。

[cuBLASの最初のビルドと実行](../nvidia/c-cpp/cublas/README.ja.md#first-build-and-run)から始めてください。このCPU-onlyの手順にGPUやPegasusは不要です。依存環境が利用できる場合に、そのCUDA/OpenACCサンプルを比較します。[ルートの対応表](../README.ja.md#what-is-implemented)から参照する6ライブラリのREADMEに、その他のサンプルのコマンドと確認事項があります。

<a id="1-prepare-dependencies-and-separate-builds"></a>

### 1. 依存環境と分離したビルドを準備する

| 実行する対象 | 必要環境 |
| --- | --- |
| cuRAND/ThrustのCPUサンプル | C++17コンパイラ。CUDAや最適化CPUライブラリは不要 |
| その他のCPUサンプル／ベンチマーク | C17/C++17、FFTW3f、選択したCBLAS/LAPACKEプロバイダ、要求される場合はoneMKL Sparse。threaded FFTWにはさらに`fftw3f_threads`が必要 |
| 直接CUDA | 対応するNVIDIA GPU/ドライバ、対象ライブラリとThrust/CCCLを含むCUDA Toolkit、互換性のあるホストコンパイラ |
| OpenACCからのライブラリ呼び出し | 同じGPUと外部CUDA Toolkit、およびNVHPCの`nvc`/`nvc++`。OpenACCで管理するデバイスデータをCUDAライブラリへ渡す |
| 正本のCPU/CUDA/OpenACC完全比較 | 上記すべて、FFTW threadedとoneMKLのCPUバックエンド、Python 3.9以降、CMake 3.20以降、single-configビルドツール |
| 既存データからのPNG出力 | Python 3.9以降とmatplotlib。GPU、コンパイラ、Pegasusへのアクセスは不要 |

まず、インストール済みツールチェーン/プロバイダの環境を整えてください。これらのコマンドは依存関係のインストールやモジュール選択を行いません。完全比較では、`CUDA_TOOLKIT_ROOT`を実際の外部Toolkitルート、`CUDA_ARCHITECTURES`をGPUに対応するCMakeアーキテクチャ値、`NVHPC_GPU_TARGET`を対応するNVHPCターゲットに設定します。`MKLROOT`と、FFTWが非既定の場所にある場合は`FFTW_ROOT`を、実際のインストール先プレフィックスでexportしてください。Pegasusのパスやアーキテクチャ値を別マシンへコピーしないでください。

以下のローカルスイート手順は、モジュールシステムの有無を問わずLinuxを対象とします。既存の依存関係/GPU検査と単一ホスト準備ツールを使い、スケジューラの資源割当は作成しません。モジュールシステムがないことは明示的に記録しますが、必須バイナリ、依存関係、GPU検査の欠如は引き続き失敗です。複数GPUのマシンでは、後述のようにGPU UUIDを明示的に選択する必要があります。マシン、キュー、モジュール、バージョンは推測しません。

```bash
: "${CUDA_TOOLKIT_ROOT:?Set your installed external CUDA Toolkit root}"
: "${CUDA_ARCHITECTURES:?Set your GPU's CMake CUDA architectures}"
: "${NVHPC_GPU_TARGET:?Set your GPU's NVHPC target}"
export NVHPC_CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT"
export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
export CUDA_PATH="$CUDA_TOOLKIT_ROOT"
export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p build results
BUILD_ROOT="$(mktemp -d "$PWD/build/reader.XXXXXX")"
RUN_DIR="$(mktemp -d "$PWD/results/local-pilot.XXXXXX")"
export CPU_CUDA_BUILD="$BUILD_ROOT/cpu-cuda"
export OPENACC_BUILD="$BUILD_ROOT/openacc"
set -o noclobber

cmake -S . -B "$CPU_CUDA_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DGPU_SUITE_BUILD_CPU=ON -DGPU_SUITE_BUILD_CUDA=ON \
  -DGPU_SUITE_BUILD_OPENACC=OFF \
  -DCMAKE_CUDA_COMPILER="$CUDA_TOOLKIT_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGPU_SUITE_CPU_BLAS_BACKEND=ONEMKL \
  -DGPU_SUITE_CPU_SPARSE_BACKEND=ONEMKL \
  -DGPU_SUITE_CPU_LAPACK_BACKEND=ONEMKL
cmake --build "$CPU_CUDA_BUILD" --target gpu_suite_partial_manifest --parallel 2

cmake -S . -B "$OPENACC_BUILD" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DGPU_SUITE_BUILD_TESTS=OFF \
  -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
  -DGPU_SUITE_BUILD_CPU=OFF -DGPU_SUITE_BUILD_CUDA=OFF \
  -DGPU_SUITE_BUILD_OPENACC=ON \
  -DCUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
  -DGPU_SUITE_NVHPC_GPU_TARGET="$NVHPC_GPU_TARGET"
cmake --build "$OPENACC_BUILD" --target gpu_suite_partial_manifest --parallel 2
```

各configureの結果一覧と`build-metadata.json`のコンパイラフラグを確認し、新しい測定を承認済み比較として扱う前に、[数値方針上の留保](#numerical-portability)を解決してください。コマンドが失敗したら停止します。マニフェストターゲットは有効なサンプル/ベンチマークをすべてビルドし、各ビルドツリーへ`partial-manifest.json`を書き込みます。configureは同じ場所へ`build-metadata.json`を書き込みます。configure成功は、すべての任意ターゲットが有効になったことを意味しません。6ライブラリの3実装をすべて比較するには、ベンチマークターゲット全18本と、そのCPUバックエンドが必要です。NVHPC、FFTW、oneMKL、バックエンドの欠如は比較の未完了であり、その系列を置き換えたり黙って削除したりする許可ではありません。

各ライブラリREADMEの**サンプルの実行**（cuBLASは**最初のビルドと実行**）に従い、有効な教材サンプルを実行します。これらの節は上でexportした2つのビルドディレクトリ変数を使用します。終了値が0以外の場合は記録し、数値出力を確認してからベンチマークへ進んでください。

<a id="2-make-a-small-measurement-check-then-choose-the-suite-settings"></a>

### 2. 小規模な測定確認を行い、スイート設定を選ぶ

cuBLASサンプルが成功した後、次の小規模CPUベンチマークで、教材掲載用ワークロードを使わずにCLI、検証、結果形式を確認します。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE \
"$CPU_CUDA_BUILD/nvidia/c-cpp/cublas/blas_cpu_bench" \
  --size 128 --alpha 1 --beta 1 --warmup 1 --repeat 2 --trials 1 \
  --scope compute --verify true --abs-tolerance 1e-12 \
  --rel-tolerance 1e-10 --cpu-backend cpu-onemkl --cpu-threads 1 \
  --cpu-backend-role production --series-role primary --cpu-parallelism threaded \
  --output "$RUN_DIR/standalone-cublas.jsonl" --format jsonl \
  2> "$RUN_DIR/standalone-cublas.stderr"
```

終了ステータスと、新規JSONLファイル内の`status="success"`、`verification_status="pass"`、指標／閾値を確認してください。この単独レコードは診断用の確認であり、完全なスイート/キャンペーン入力ではありません。後述のランナーのrawファイルに**混ぜないでください**。GPU smoke確認では、対応ベンチマーク実行ファイルがCPUバックエンドオプションを除く同じsize/スコープ/検証オプションを受け付けます。ライブラリREADMEに従い、呼び出しごとに別の出力ファイル名を使ってください。

続いて、実際のスイート入力を準備します。

```bash
python3 tools/merge_manifests.py --output "$RUN_DIR/executables.json" \
  "$CPU_CUDA_BUILD/partial-manifest.json" \
  "$OPENACC_BUILD/partial-manifest.json"
cp configs/pilot.json "$RUN_DIR/config-input.json"
CONFIG_INPUT="$RUN_DIR/config-input.json"
MANIFEST="$RUN_DIR/executables.json"
```

`pilot.json`の試行数は1ですが、**サイズとcompute repeatは教材掲載用キャンペーンと同じ大きな値**です。少ないメモリで実行するsmoke用presetではありません。自分の実験に小さいケースが必要なら、実行前に新規作成したconfig-入力コピーをtext editorで編集します。リポジトリの正本ファイルと過去のrunは変更しないでください。測定開始後はこの保存コピーを不変とし、後から変更するには新しいrunディレクトリ/IDが必要です。

最初の具体的な**3実装スイート診断**では、`run_mode="smoke"`、`cpu_threads=1`とし、`benchmarks.cublas.enabled=true`だけを有効にします。他の5個の`enabled`はfalseにしますが、オブジェクト自体は残します。cuBLASは最初のケースだけを残して`size`を128にし、両スコープを`warmup=1`、`repeat=1`、`trials=1`にします。`alpha=beta=1`、3つのprimary系列、oneMKLの有効スレッド数null、既存許容誤差は維持します。予定行数は、1ケース×2スコープ×3実装×1試行の6です。以下の検証・集計はできますが、**6枚の教材掲載用図は生成できません**。後に完全なスイープを行う場合は、新しいrunディレクトリ／設定から始めてください。問題サイズだけを変えるためにビルドし直す必要はなく、変更のないビルドツリーは再利用できます。

| 入力コピー内のJSON位置（後で実効設定としてserializeする） | 意味と適切な使い方 |
| --- | --- |
| `run_mode` | 縮小診断は`smoke`、予備スイープは`pilot`。`production`には既知かつcleanなGit provenanceとReleaseバイナリが必要 |
| `benchmarks.<library>.cases[i].parameters` | ワークロード sizeを明示する。例：cuBLAS `size=128`、cuFFT `nfft=256,batch=8`、cuSPARSEの平方数`size=4096`、cuSOLVER `size=128,nrhs=16`、cuRAND/Thrust `size=1048576`は診断候補であり、測定に基づく性能上の推奨値ではない |
| `cases[i].scopes.compute`と`.end-to-end` | `warmup`、`repeat`、`trials`。小規模確認では`1,1,1`を使える。cuSOLVER repeatは必ず1、教材掲載用 E2E repeatも1を維持 |
| `cpu_threads` | 割当範囲内の要求スレッド数。手順3で設定する環境も制御する |
| `benchmarks.<library>.series` | 明示的なバックエンド/role/スレッドメタデータ。primary CPU/CUDA/OpenACCエントリを維持する。FFTWの有効スレッド数は設定したスレッド APIに従い、oneMKLは観測できなければnull、cuRAND/Thrustは有効スレッド数1を維持 |
| `output_format` | この手順では`jsonl`を維持する。検証／集計はCSVも受け付けるが、現在の描画raw-入力 loaderはJSONLが必要 |
| `verification`、`precision`、アルゴリズムパラメータ | 承認済みの数学・精度・検証契約を維持する。精度文字列はアルゴリズムを再コンパイルするswitchではない。高速で成功する結果を得るためだけに許容誤差を緩めない |

設定には6ライブラリすべてのオブジェクトと両スコープが必要です。明示的な部分診断では、ライブラリ全体を`enabled=false`にできますが、必須のprimary実装だけを削除して完全比較と呼ぶことはできません。plotterはさらに厳しく、各ライブラリに異なる3ケース、両スコープ、全3実装を要求します。1ケースまたは単一ライブラリのsmokeは図の入力にはなりません。CPUバックエンド名はデータから取得し、cuSOLVERの目盛りは教材掲載用 sizeを仮定せず実際のケースに従います。

編集後、正本のJSON serializerで不変のeffectiveコピーを作成します。入力コピーは整形済みでも構いませんが、effectiveコピーはキャンペーン準備が要求する決定的な表現でなければなりません。

```bash
CONFIG="$RUN_DIR/effective-config.json"
PYTHONPATH=tools python3 - "$CONFIG_INPUT" "$CONFIG" <<'PY'
import sys
from pathlib import Path
from gpu_suite.config import load_config
from gpu_suite.strict_json import dump_bytes
with Path(sys.argv[2]).open("xb") as output:
    output.write(dump_bytes(load_config(sys.argv[1])))
PY
```

<a id="3-capture-the-real-runtime-environment-and-launch-one-local-block"></a>

### 3. 実際の実行環境を取得し、ローカルブロックを1つ実行する

同じツールチェーン設定のシェルを使い続けてください。これらのexportは要求値であり、実際のCPU使用率の測定値ではありません。アプリケーションのOpenMPループを追加するものでもありません。

```bash
CPU_THREADS="$(PYTHONPATH=tools python3 -c \
  'import sys; from gpu_suite.config import load_config; print(load_config(sys.argv[1])["cpu_threads"])' \
  "$CONFIG")"
export OMP_NUM_THREADS="$CPU_THREADS" OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_DYNAMIC=FALSE MKL_NUM_THREADS="$CPU_THREADS" MKL_DYNAMIC=FALSE
export MKL_THREADING_LAYER=INTEL OPENBLAS_NUM_THREADS="$CPU_THREADS"
CUDA_TOOLKIT_VERSION="$(PYTHONPATH=tools python3 -c \
  'import sys; from gpu_suite.strict_json import load; print(load(sys.argv[1])["cuda"]["toolkit_version"])' \
  "$CPU_CUDA_BUILD/build-metadata.json")"
MODULE_OPTIONS=(--no-module-system)
# If your shell actually uses modules, replace that declaration with:
# module list > "$RUN_DIR/module-list.txt" 2>&1
# MODULE_OPTIONS=(--module-list "$RUN_DIR/module-list.txt")
GPU_SELECTION_ARGS=()
RUNTIME_HASH="$(python3 jobs/pegasus/collect_runtime_environment.py \
  --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  "${MODULE_OPTIONS[@]}" \
  --cuda-toolkit-root "$CUDA_TOOLKIT_ROOT" \
  --cuda-toolkit-version "$CUDA_TOOLKIT_VERSION" \
  --nvhpc-cuda-home "$NVHPC_CUDA_HOME" \
  --require-ldd --require-gpu-tools \
  --output "$RUN_DIR/runtime-environment.json" \
  --evidence-output "$RUN_DIR/runtime-environment-evidence.json")"
```

この収集コマンドは`jobs/pegasus`にありますが、jobを投入せず、アカウント/キューも必要としません。`--no-module-system`は明示的な利用者宣言と空のidentityリストを記録し、モジュールコマンドの出力／ハッシュを捏造しません。モジュールを使う場合は、受理可能な実際のリストを指定してください。失敗または空のモジュールコマンドを、自動的にモジュールを使わないと再分類することはありません。8個のCPU変数すべて、整合するToolkitルート/バージョン、正規化された絶対検索パス、解決済みバイナリ依存関係は引き続き必須です。コレクタが0以外で終了したら、**ここで停止してください**。2つの`--require-*` フラグにより依存関係とGPUランタイムの確認を必須とします。一方、任意のコンパイラ／パッケージのバージョン検査はunavailableと記録されることがあります。利用できないバージョンを観測済みと主張せず、この違いを保ってください。モジュール識別情報や架空のハッシュを作ったり、所定のランタイム文書の代わりに任意の`env` dumpをハッシュ化したりしないでください。

複数GPUマシンでは、収集前に`nvidia-smi`で観測したUUIDを選び、その正確なUUIDを`CUDA_VISIBLE_DEVICES`としてexportします。上記の空配列初期化の後で、`GPU_SELECTION_ARGS=(--gpu-uuid "$CUDA_VISIBLE_DEVICES")`を設定してください。ローカルworkerは論理デバイス 0を使います。local準備はGPUを推測せず、曖昧または不一致なvisibilityを拒否します。

保存するランタイム文書には、ビルドメタデータとバイナリ/依存関係ハッシュを埋め込みます。コマンドはその正確なSHA-256を`RUNTIME_HASH`として出力します。起動時にマニフェスト内の絶対実行パスとハッシュを使うため、2つのビルドツリーは変更しないでください。次に、実際のlocal run/wave/node provenanceを準備します。観測したhostname、利用可能なら`lscpu`のCPU識別情報、および独立したGPU照会を使います。スケジューラフィールドはnullのままで、PBS識別子やjob状態を捏造しません。`PROVENANCE_ARGS`が空なのは、この手順が想定するclean-ビルドの場合だけです。

```bash
RUN_ID="$(basename "$RUN_DIR")"
SYSTEM_LABEL=local-nvidia
PROVENANCE_ARGS=()
mkdir "$RUN_DIR/campaigns"
NODE_DIR="$(python3 jobs/pegasus/prepare_wave.py --local \
  --result-root "$RUN_DIR/campaigns" --run-id "$RUN_ID" --wave 0 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  --runtime-environment "$RUN_DIR/runtime-environment.json" \
  --runtime-environment-evidence "$RUN_DIR/runtime-environment-evidence.json" \
  --system-label "$SYSTEM_LABEL" "${GPU_SELECTION_ARGS[@]}" "${PROVENANCE_ARGS[@]}")"
python3 tools/run_suite.py --config "$CONFIG" --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  --run-id "$RUN_ID" --system-label "$SYSTEM_LABEL" --wave 0 --node-index 0 \
  --runtime-environment-sha256 "$RUNTIME_HASH" "${PROVENANCE_ARGS[@]}" --dry-run \
  > "$RUN_DIR/resolved-plan.json"
```

実行前に、プランが選んだバイナリ、パラメータ、CPUバックエンド、順序、`prerequisite_failure` フィールドを確認してください。このコマンドは成果物を確認しますが、それ自体を実行しません。dirtyビルドには、[ソース provenance（英語）](RESULT_SCHEMA.md#executables-manifest-and-source-provenance)で定める完全なソース/diffハッシュが1つ必要です。Gitメタデータのないアーカイブを、cleanなproductionビルドと説明してはいけません。この手順はcleanなビルドメタデータを想定します。そうでない場合は、**準備の前に**、適切な`--git-diff-sha256`または`--source-snapshot-sha256`と、正本のビルド時ハッシュを`PROVENANCE_ARGS`に設定し、両ランナーコマンドへ同じ値を渡してください。架空の値を使ったり、バイナリのビルド後に変更したソースをハッシュ化して代用したりしないでください。

```bash
python3 tools/run_suite.py --config "$CONFIG" --manifest "$MANIFEST" \
  --build-metadata "$CPU_CUDA_BUILD/build-metadata.json" \
  --build-metadata "$OPENACC_BUILD/build-metadata.json" \
  --run-id "$RUN_ID" --system-label "$SYSTEM_LABEL" --wave 0 --node-index 0 \
  --runtime-environment-sha256 "$RUNTIME_HASH" "${PROVENANCE_ARGS[@]}" \
  --output "$NODE_DIR/raw-results.jsonl" \
  2> "$NODE_DIR/runner.stderr"
RUNNER_RC=$?
printf 'runner exit status: %s\n' "$RUNNER_RC"
RAW_FILES=("$NODE_DIR/raw-results.jsonl")
PLOT_OPTIONS=(--node-metadata "$NODE_DIR/node-metadata.json")
AGGREGATE_OPTIONS=()
ANALYSIS_DIR="$RUN_DIR/analysis"
mkdir "$ANALYSIS_DIR"
python3 jobs/pegasus/node_tools.py classify --raw "${RAW_FILES[0]}" \
  --node-metadata "$NODE_DIR/node-metadata.json" \
  --output "$ANALYSIS_DIR/node-classification.json"
```

ランナーは実装を交互に実行し、rawファイルを排他的に新規作成します。run/wave/nodeメタデータはランナーではなく準備コマンドが作成済みです。このlocal経路はjob完了／回収状態を捏造せず、テレメトリや複数ノード割当も実装しません。これは**単一ノードのpilot解析**であり、教材の6ノード・2 waveの結果ではありません。失敗時はraw行と標準エラー出力を保持し、同じ出力パスに再試行しないでください。次の検証手順で失敗レコードを調べ、成功したかのように性能図へ進まないでください。完全なPegasusランチャ、メタデータ、回収ワークフローには[専用手順（英語）](PEGASUS_EXECUTION.md)を使います。他システムには対応するランチャが必要であり、架空のPBS識別子では代用できません。

<a id="validate-and-aggregate-a-completed-run"></a>

## 完了したrunを検証・集計する

入力は、上のlocal手順または下の保存データ手順で設定した`CONFIG`、`MANIFEST`、Bashの`RAW_FILES`配列です。`ANALYSIS_DIR`は、新規に作成済みの出力ディレクトリである必要があります。単独実行診断ファイルをglobで取り込んだり、設定、バイナリ、ランタイムハッシュ、run IDを混在させたりしないでください。

```bash
python3 tools/validate_results.py --config "$CONFIG" --manifest "$MANIFEST" \
  --report-output "$ANALYSIS_DIR/validation-report.json" "${RAW_FILES[@]}"
python3 -m json.tool "$ANALYSIS_DIR/validation-report.json"
```

完全比較には、終了値0、`validation_status="pass"`、成功statusだけであることを要求します。rawの`verification_status`、指標、閾値を確認し、要求した数値検証の成功が示されていなければなりません。validatorは、設定されたケース/試行、順序、マニフェスト/config provenanceも確認します。offlineのマニフェスト確認には、元の実行パスが存在する必要はありません。ただし、productionランチャのノードローカルランタイム/テレメトリ確認の代わりにはなりません。検証が失敗したらreportとrawデータを保持して原因を調べ、成功比較としての処理は停止します。

```bash
python3 tools/aggregate.py --config "$CONFIG" "${AGGREGATE_OPTIONS[@]}" \
  --output "$ANALYSIS_DIR/aggregate-results.jsonl" \
  --metadata-output "$ANALYSIS_DIR/aggregate-metadata.json" "${RAW_FILES[@]}"
AGGREGATE_INPUT="$ANALYSIS_DIR/aggregate-results.jsonl"
```

出力はブロック、wave、wave間、および別ラベルの探索的レコードを含みます。描画が選ぶのはprimary wave間の実行時間レコードであり、speedupレコードや全試行を再poolした結果ではありません。aggregateの標本数、失敗数、`aggregate_status`を確認してください。1 wave/ブロックでも中央値は得られますが、四分位数や変動に関する主張にはデータが不足し得ます。集計実装は失敗数を保持しつつ、failed/skipped/nonfinite行を数値標本から除外します。遅い成功ノードを削除したり、rawレコードを修繕したりしないでください。

<a id="figures-from-your-own-measurements"></a>

## 自分の測定から図を作る

この手順の目標は、教材マシンと同じ実行時間や順位ではなく、**同じ方法と2パネル構成**です。現在のplotterには次が必要です。

- 1つのrun/ランタイム provenanceに属するprimary wave間レコード。
- cuFFT、cuBLAS、cuSPARSE、cuSOLVER、cuRAND。ツールとしてThrustは任意ですが、この教材の6ライブラリ完全結果には必要です。
- 両スコープで一致するちょうど3ケースと、CPU/CUDA/OpenACCの系列。
- 各primary CPU系列の実際のCPUバックエンド識別情報。
- 同一ライブラリバージョンを報告する、成功したThrust CUDA/OpenACC raw行。

GPU名は成功raw行から取得します。local手順では`--node-metadata`が観測したCPU名を提供し、欠けた識別情報はunknownと表示します。要求CPU数と報告された有効CPU数を区別し、既知のバックエンド IDは読みやすい名前に、その他のIDはそのまま表示します。異なるマシンモデルの混在は、1つのラベルに隠さず拒否します。cuSOLVERの目盛りは実際の3サイズを使い、教材のケースでは`4K/8K/12K`を保持します。通常の凡例は同じ2パネルの下に縦に並べ、長いラベルは折り返します。明示的な教材表示モードでは、承認済みの3列凡例を保持します。

識別情報を観測できない場合は、厳密な[表示設定（英語）](RESULT_SCHEMA.md#plot-display-configuration)を`--display-config`で指定します。指定値はplotメタデータで`user-specified`と記録し、観測した識別情報やrun/ランタイム provenanceと矛盾できません。この手順では`thread_label_mode="requested-and-effective"`を使います。実際の値で設定を作成した後にだけ、`PLOT_OPTIONS=(--display-config "$RUN_DIR/plot-display.json" --node-metadata "$NODE_DIR/node-metadata.json")`を設定してください。凡例を見かけ上完全にするためだけに識別情報を書かないでください。`--system-label`はキャンペーン識別子であり、ハードウェア情報の上書きではありません。

これらの確認後、[図の生成と確認](#render-and-inspect-figures)へ進んでください。自分の測定には、次の保存データ用の参考手順は不要です。

<a id="regenerating-the-teaching-figures-from-saved-data"></a>

## 保存データから教材の図を再生成する

**今回のコード公開には含みません：**測定データと作成済みローカル配布アーカイブは、追跡対象ソースの外に非公開で保持しています。参照できるRelease／データ URLはありません。以下のコマンドは、必要な保存入力を別途保有する読者向けの条件付き参考手順です。ソースのcheckoutだけでは入力は得られません。通常の読者向け手順は[自分の測定から図を作る](#figures-from-your-own-measurements)です。

保存キャンペーンは`publication-benchmark-20260716T025037Z`で、測定commitは`916a8bda22f6188aa995d21954ddb1d6de2f3cf3`です。承認済みrendererのcommitは`4d3c2333d2a58aa3fc2a8a82d43c403d2131343f`で、この2版の間で変更したのは描画とそのテストだけです。更新済み読者向けツールは、バンドルの明示的な`plot-display.json`を与えた場合に、その表示を維持します。バンドル READMEに記録されたコード版と作業ツリー/ソースハッシュ、および新しく編集した設定ではなく**保存済み**実効設定を使ってください。[保存根拠（英語）](VALIDATION_REPORT.md#saved-execution-evidence-reviewed-for-publication)と[教材掲載用条件](BENCHMARK_PROTOCOL.ja.md#canonical-publication-configuration)では、1 waveあたり6ノード、2 wave、5試行、3ケース、両スコープ、全6ライブラリ（6,480行）を定めています。6つの出力ファイル名は、[教材掲載用図](BENCHMARK_PROTOCOL.ja.md#publication-figures)の6ライブラリ名に直接対応します。

| 目的 | 必要データと、現在のツールが実際に読むもの |
| --- | --- |
| 既存要約から6枚のPNGを描画 | `publication-output/aggregate-results.jsonl`、同じキャンペーンのJSONL raw根拠、承認済みラベル用の`plot-display.json`。CLIは常にraw入力を要求し、Thrustは成功したCUDA/OpenACCの`library_version`と、run/ランタイム識別情報の一致を確認する |
| 検証と要約の再構築も行う | 元の全12ノードの`raw-results.jsonl`、バイト一致の`effective-config.json`、`executables-manifest.json`。検証ではマニフェストをメタデータとして読み、offline手順に元のGPUバイナリは不要 |
| 測定条件を説明・監査する | `run-metadata.json`、`runtime-environment.json`、wave/nodeメタデータ、各ノードのCPUハードウェア根拠（保存キャンペーンでは`telemetry/cpu-telemetry.txt`）、checksum一覧。これらはprovenance/キャプションの根拠であり、`plot.py`の位置引数ではない。nodeメタデータだけにはCPUモデルは含まれない |
| 有用な参考情報で、必須描画入力ではないもの | 保存済みvalidation/aggregate/plotメタデータと承認済みPNG。PNGだけでは再生成入力にならない |

完了済みの非公開データ再生成では、rendererのバージョン確認を満たすThrust 2行だけではなく、元の12 rawファイルすべてと、aggregate、実効設定、マニフェスト、provenance/CPUハードウェア根拠を使いました。PNGを描画するだけなら、完全なビルドログ、連続GPUテレメトリ、PBSログ、バイナリ配布は不要です。原本成果物は保持しており、作成済みアーカイブは非公開の検証成果物であって、今回のコード公開の配布物ではありません。

以下の条件付きコマンドはリポジトリのルートのBashで実行し、`DATA_DIR`を、次の相対配置を持つ別途提供された保存データのルートに設定します。元の絶対実行パスは歴史的なマニフェスト値であり、再生成する利用者が同じパスを持つ必要はありません。この再生成手順は、測定手順のシェル変数に依存しません。

```bash
: "${DATA_DIR:?Separately supplied saved data is required; not included with source}"
(cd "$DATA_DIR" && shasum -a 256 -c SHA256SUMS)
CONFIG="$DATA_DIR/effective-config.json"
MANIFEST="$DATA_DIR/executables-manifest.json"
RAW_FILES=("$DATA_DIR"/waves/*/nodes/*/raw-results.jsonl)
AGGREGATE_INPUT="$DATA_DIR/publication-output/aggregate-results.jsonl"
PLOT_OPTIONS=(--display-config "$DATA_DIR/plot-display.json")
AGGREGATE_OPTIONS=(--no-pooled)
test -f "$CONFIG" && test -f "$MANIFEST" && test -f "$AGGREGATE_INPUT"
test "${#RAW_FILES[@]}" -eq 12
mkdir -p results
ANALYSIS_DIR="$(mktemp -d "$PWD/results/replay.XXXXXX")"
```

ファイル／checksumが欠ける、または確認に失敗した場合は停止してください。[検証](#validate-and-aggregate-a-completed-run)で提供されたrawキャンペーンを確認します。提供aggregateから直接描画する場合は、上で設定した`AGGREGATE_INPUT`を維持して描画へ進みます。集計の再実行は任意です。要約を再構築する場合、集計コマンドは`ANALYSIS_DIR`に新しいファイルを書き、`AGGREGATE_INPUT`をその新ファイルへ設定し直します。バンドルを上書きしないでください。`--no-pooled`は保存済みの2,700-行 aggregateと一致します。既定ではさらに180行のラベル付き探索的pooledレコードを出力しますが、図に使うブロック/wave/wave間値は変わりません。どちらのoffline手順も、GPUやPegasusへのloginは不要です。

<a id="render-and-inspect-figures"></a>

## 図の生成と確認

自分の測定では、上の注記／系列の条件を満たしてから実行してください。条件付きの保存データ手順でも同じコマンドを使いますが、別途必要な入力条件を満たした場合に限ります。入力は既存の`AGGREGATE_INPUT`と元のJSONL `RAW_FILES`です。以下の2つの出力先は、まだ存在していてはいけません。

```bash
mkdir "$ANALYSIS_DIR/plots"
python3 tools/plot.py "$AGGREGATE_INPUT" \
  --output-directory "$ANALYSIS_DIR/plots" \
  --metadata-output "$ANALYSIS_DIR/plot-metadata.json" \
  --raw-results "${RAW_FILES[@]}" "${PLOT_OPTIONS[@]}"
python3 -m json.tool "$ANALYSIS_DIR/plot-metadata.json"
ls "$ANALYSIS_DIR/plots"
```

`matplotlib.status="rendered"`、予定する6個すべての`generated_files`、実際のPNGを確認してください。終了値0だけでは不十分です。matplotlibが利用できない場合、ツールは`unexecuted`と記録し、PNGなしで0で終了することがあります。画像ビューアで図を開き、2つのタイトル、下部の共通凡例、ハードウェア/バックエンド/スレッドラベル、各実装3点、経過時間の単位、2進数表記の目盛りを確認します。[解釈ガイド](BENCHMARK_PROTOCOL.ja.md#reading-the-figures-and-applying-the-results)に従ってください。matplotlib/フォント/プラットフォームのバージョンが違うと、ラスターの見た目は変わり得ます。同じデータ／配置でも、PNGのバイト一致は保証しません。rendererのrevisionとplotメタデータに記録されたmatplotlibバージョンを残してください。

<a id="known-limitations-and-out-of-scope-work"></a>

## 既知の制限と今回の対象外

マシンラベル、可変のcuSOLVER目盛り、明示的なモジュールを使わない収集、単一ホストの非PBS run/wave/node準備は実装済みです。以下の制限は引き続き明示します。追加基盤、GPU診断、保存データ配布は、今回の[コード公開範囲](../README.ja.md#publication-scope)の前提条件ではありません。

| 項目 | 現在の制限 |
| --- | --- |
| ローカル読者手順のGPU実行 | モジュール/PBSのないLinux NVIDIA/NVHPC環境で、GPU読者ワークフロー全体は未実行。CPU-onlyフィクスチャや保存データの再生成で、その環境の動作を保証することはできない |
| Pegasus以外の複数ノード制御とテレメトリ | 単一ホスト手順には未実装で、今回の公開に向けた開発範囲外。架空のスケジューラ状態は生成しない |
| 継承されたNVHPC Releaseフラグの数値方針確認 | 記録された`-fast`条件と不明な当時の実行時FPU状態を引き続き開示し、方針適合済みとはしない。今回の公開では数値プロファイルの導入やキャンペーン再測定を行わず、保存成果物は変更しない |
| 教材データの再生成 | 保存入力と作成済みアーカイブは今回の公開に含まない。参考手順には別途提供されるデータが必要であり、通常の読者手順では自分で測定する |

[検証報告（英語）](VALIDATION_REPORT.md#reader-tools-and-offline-replay-validation)では、ローカルツールテストとoffline再生成を、GPU実行や公開操作と区別しています。
