# GPU Library Suiteベンチマークプロトコル

[English](BENCHMARK_PROTOCOL.md) | [日本語](BENCHMARK_PROTOCOL.ja.md)

<a id="authority-and-scope"></a>

## 正本としての範囲

本書は、ベンチマークのコマンドライン動作、ワークロードスイープ、測定境界、warm-upと状態復元、検証、CPUバックエンドの分類、実行順序、統計集計を定めます。教材ファイル名、サンプルの既定値、数学的定義は[`PROJECT_SPECIFICATION.md`（英語）](PROJECT_SPECIFICATION.md)、シリアライズとフィールドの型は[`RESULT_SCHEMA.md`（英語）](RESULT_SCHEMA.md)、システムの起動とテレメトリの詳細は[`PEGASUS_EXECUTION.md`（英語）](PEGASUS_EXECUTION.md)が定めます。

<a id="general-measurement-principles"></a>

## 測定の基本原則

- CPU、直接CUDA、OpenACCは、同じ数学的問題を、同じ入力、精度、問題サイズ、スコープ、repeat数で解きます。
- 全実装で`clock_gettime(CLOCK_MONOTONIC, ...)`に基づく同じ経過時間計測の共通関数を使います。
- 試行系列ごとに`clock_getres(CLOCK_MONOTONIC, ...)`を記録します。
- CUDA Eventsは比較の主要タイマーには使いません。
- warm-up、検証、結果シリアライズ、ファイルI/Oは測定区間外です。
- raw試行をすべて保存します。ベンチマーク実行ファイル内でノード間集計や外れ値除去は行いません。
- 1試行が短すぎて信頼できる測定にならない場合は`repeat`を増やし、数学的問題を黙って変更しません。
- 同じベンチマーク／問題／スコープでは、CPU、CUDA、OpenACCが同じrepeatを使います。スイートランナーは不一致を拒否します。
- 検出したハードウェアに応じて、問題サイズ、アルゴリズム、精度、数値モードを選択しません。
- メモリ不足は明確な診断と失敗を示す行で報告します。
- ベンチマーク実行ファイルは単一ノード・単一プロセスです。複数ノードでの反復は、1ノード1プロセスのランチャが担当します。

<a id="command-line-interface"></a>

## コマンドラインインターフェース

<a id="common-options"></a>

### 共通オプション

すべてのベンチマークが次に対応します。

| オプション | 意味 |
| --- | --- |
| `--size` | 後述の、ライブラリ固有の主要問題サイズ |
| `--warmup` | 0以上のwarm-up回数 |
| `--repeat` | 各試行内の演算またはパイプラインの正の反復回数 |
| `--trials` | raw試行数。正の値 |
| `--scope` | `compute`または`end-to-end` |
| `--verify true\|false` | 検証の有効／無効を明示 |
| `--output <path>\|-` | 排他的な結果出力パス、または機械可読標準出力 |
| `--format csv\|jsonl` | 出力シリアライズ。実効設定またはCLIで選択する |
| `--device` | 0以上のCUDAデバイスインデックス |
| `--run-id` | 測定キャンペーン識別子 |
| `--system-label` | 利用者が指定する移植可能なシステムラベル |
| `--node-index` | ランナーが渡すMPI rank |
| `--wave` | ランナーが渡す0以上のキャンペーン wave |
| `--seed` | 乱数を使うワークロードの明示的なシード |
| `--cpu-threads` | 要求CPUスレッド数 |
| `--cpu-threads-effective` | 観測／設定された正の有効CPUスレッド数。不明なら省略 |
| `--cpu-backend-role` | 設定が定めるCPU role：`production`または`reference` |
| `--series-role` | 設定が定める系列role：`primary`または`auxiliary` |
| `--cpu-parallelism` | 設定／観測によるCPU実行種別：`serial`、`threaded`、`unknown` |
| `--implementation-order <list>` | ランナーが計算するカンマ区切りリスト。例：`cpu,cuda,openacc` |
| `--abs-tolerance` | 実効設定から与える0以上の絶対許容誤差 |
| `--rel-tolerance` | 実効設定から与える0以上の相対許容誤差 |
| `--help` | 使い方とオプションの制約 |

`--verify`が受け付けるのは小文字の`true`または`false`だけです。`--format`は`csv`または`jsonl`だけを受け付けます。`--implementation-order`は、後述するCPU/CUDA/OpenACCの正確な6通りのカンマ区切り順列のいずれかを受け付けます。

`--node-index`、`--wave`、`--implementation-order`は実行コンテキストのオプションです。スイート実行中にこれらを設定するのは`run_suite.py`だけであり、所定の順序規則と矛盾する値は拒否します。

<a id="output-contract-and-writer-ownership"></a>

### 出力契約と書込みの責任

`--output -`では、ベンチマーク実行ファイルは機械可読の結果だけを標準出力へ出力します。人間向けメッセージ、警告、進捗行、診断はすべて標準エラー出力へ出力します。標準出力への人間向け内容の混入は出力エラーです。

`--output <path>`では、新しいファイルを排他的に作成します。パスがすでに存在する場合は必ず失敗し、上書きや暗黙の追記をしてはいけません。

スイート実行中、`run_suite.py`は必ず各ベンチマークを`--output -`で起動します。ノード単位の`raw-results.csv`または`raw-results.jsonl`を書き込むのはランナーだけです。ベンチマーク実行ファイルがノードrawファイルへ直接追記することはありません。ランナーは次を満たす必要があります。

- 行を受理する前に、子プロセスの標準出力をスキーマ検証する。
- CSVヘッダをちょうど1回書く。
- 有効な各行を新しいノードrawファイルへ追記する。
- 人間向け標準出力内容をすべて拒否する。
- 検証済みの完了試行をすべて保持する。
- 子プロセスのcrash、signal終了、不正出力、出力欠落の後、最初の未解決試行に対し、スキーマに適合したattempted失敗を示す行を1つ生成する。
- その失敗によって開始できなかった後続試行には、スキーマに適合したunattempted skipped行を生成する。
- ノードrawファイルを排他的に新規作成し、既存ファイルを上書きしない。

ランナーは重複または範囲外の試行インデックスを拒否し、予定した各インデックスをちょうど1回だけ記録します。検証済みの完了行を、生成した代替行に置き換えることはありません。

`run_suite.py --dry-run`は、解決済み設定、選択したマニフェストエントリ、実装／サイズ順、argv配列を検証し、1つの決定的なJSON文書として表示します。dry runはディレクトリ、メタデータ、結果、ログファイルを作らず、ベンチマーク子プロセスも起動しません。

<a id="library-specific-options-and---size"></a>

### ライブラリ固有オプションと`--size`

<a id="cufft"></a>

#### cuFFT

- オプション：`--batch`、`--transform`、`--cpu-backend`。
- `--rel-tolerance`はDC相対誤差、`--abs-tolerance`はnon-DC最大絶対誤差に適用します。
- `--size`は`nfft`です。
- `--batch`は`--size`と併用できます。
- 初期実装で受け付ける値は`--transform c2c-forward`と`--cpu-backend cpu-fftw-serial|cpu-fftw-threaded`です。

<a id="cublas"></a>

#### cuBLAS

- オプション：`--m`、`--n`、`--k`、`--alpha`、`--beta`、`--cpu-backend`。
- `--size N`は`m = n = k = N`を意味します。
- `--size`と`--m`、`--n`、`--k`は排他的です。
- 個別の次元を指定する場合は、3つすべてが必要です。
- 正方行列のスイープでは`problem_size = N`を記録します。
- 非正方のケースでは`problem_size = null`とし、`m`、`n`、`k`を`parameters`に保存します。

<a id="cusparse"></a>

#### cuSPARSE

- オプション：`--nx`、`--ny`、`--alpha`、`--beta`、`--cpu-backend`。
- `--size`は未知数の総数`N = nx * ny`です。
- `--size`を使う場合、Nは完全平方数で、`nx = ny = sqrt(N)`です。
- `--size`と`--nx`、`--ny`は排他的です。
- 次元を指定する場合は、両方が必要です。
- `PROJECT_SPECIFICATION.md`の厳密な非零要素数の式を使い、`problem_size = nx * ny`と`secondary_size = nnz`を記録します。

<a id="cusolver"></a>

#### cuSOLVER

- オプション：`--nrhs`、`--cpu-backend`。
- `--size`は密行列の次数nです。
- `secondary_size = nrhs`を記録します。

<a id="curand"></a>

#### cuRAND

- オプション：`--generator`、`--distribution`、`--offset`、`--order`。
- 統計計算の値は実効設定から、`--sigma-multiplier`、`--expected-mean`、`--expected-second-central-moment`として明示的に渡します。
- `--size`は生成要素数です。
- 初期実装で受け付ける値は`--generator pseudo-default`、`--distribution uniform-double`、`--order default`です。

<a id="thrust"></a>

#### Thrust

- オプション：`--operation`、`--cpu-backend`。
- `--size`は処理要素数です。
- 初期実装で受け付ける演算は`--operation transform-reduce-square-sum`です。

曖昧、不完全、矛盾する次元引数はエラーです。あるオプションが他のオプションを黙って上書きすることはありません。

<a id="dynamic-size-and-overflow-contract"></a>

### 動的サイズとoverflowの契約

実行時の次元はすべて、メモリ確保または従来のライブラリAPI整数型への変換の前に検証します。ベンチマークは、CLIの符号なし表現から`int`と`size_t`への検査付き変換、次元の積、CSR行-オフセット数などの加算、要素数からバイト数への計算を検査します。cuFFTのlength/batch、cuBLASの次元、現在の32-ビットインデックスモードにおけるcuSPARSEの`n`/`nnz`、cuSOLVERの`n`/`nrhs`/作業領域数は、使用するすべてのAPI型に収まる必要があります。cuRAND/Thrustの個数は、`size_t`と選択したホストコンテナの`max_size()`の両方に収まる必要があります。

次元、インデックス数、バイト数のwraparoundや暗黙の切捨ては許しません。範囲外の要求には、API名を含む診断と、実行開始前か開始後かに応じたスキーマに適合したなprerequisiteまたはベンチマーク失敗を示す行を生成します。メモリ確保・長さに関するC++例外は明示的に処理し、有効行をまだ出力できる場合に、結果を残さずプロセスを終了させません。

<a id="canonical-publication-configuration"></a>

## 正本の教材掲載用設定

各ベンチマーク設定は正規化されたケースを含みます。ケースは1つの`parameters` オブジェクトと、`compute`／`end-to-end`を分けた`scopes` オブジェクトを持ちます。各スコープエントリは`warmup`、`repeat`、`trials`を含み、実装側でこれらを上書きできません。

同じベンチマーク／正規化された問題／スコープでは、CPU、直接CUDA、OpenACCの値を同一にします。computeとend-to-endの設定は独立で、repeatやwarm-upが同じである必要はありません。cuSOLVERはすべてのスコープで`repeat = 1`を要求します。

正本のpilot／production設定は、同じ教材掲載用ワークロード、size、repeatを使います。

| ベンチマーク | 問題サイズ | compute repeat |
| --- | --- | --- |
| cuFFT | `nfft=[256,4096,16384]`, `batch=4096` | `[5197,328,83]` |
| cuBLAS | `size=[512,2048,4096]` | `[2184,133,18]` |
| cuSPARSE | `size=[65536,1048576,4194304]` | `[5776,1520,206]` |
| cuSOLVER | `size=[4096,8192,12288]`, `nrhs=16` | `[1,1,1]` |
| cuRAND | `size=[1048576,16777216,67108864]` | `[640,173,54]` |
| Thrust | `size=[1048576,16777216,67108864]` | `[1932,532,167]` |

すべてのスコープは`warmup=1`です。computeは表のケースごとのrepeat、end-to-endは常に`repeat=1`を使います。`configs/pilot.json`は1 raw試行、`configs/benchmark.json`は5 raw試行です。どちらも48 CPUスレッドを要求します。cuFFTの教材掲載用系列はthreaded FFTWだけです。

全6ライブラリでは、1 pilotブロックの予定raw行数は108です。6ノード・2 waveのproduction設計では6,480行です。pilotでCUDA/OpenACCのThrustバージョンが異なると判明した場合、比較できない系列を混ぜずにThrustを除外します。その場合の行数は90と5,400です。このように明示的に縮小したキャンペーンは5ライブラリ比較であり、全6ライブラリの再現成功ではありません。不一致の根拠を保持し、failed行を削除したり、図から1実装を黙って省略したりしないでください。

compute repeatは測定の増幅だけを目的とし、`elapsed_sec`は1回のライブラリ演算の経過時間を表し続けます。Pegasus pilot後、人間が正本値を一度変更できるのは、OOM、検証失敗、極端なwalltime、明らかに不十分な測定区間の場合だけです。repeatの自動探索や別の正本設定を追加してはいけません。

<a id="trial-attempt-failure-and-skip-behavior"></a>

## 試行の開始・失敗・skip

開始後に失敗した試行は`attempted=true`、`status=failure`です。先行する致命的失敗で開始できなかった試行は、`attempted=false`、`status=skipped`です。実行ファイルの欠如などのprerequisite不成立でも、unattempted skipped行を生成します。フィールドとnullの厳密な規則は`RESULT_SCHEMA.md`が定めます。

復旧可能な検証失敗の後は、完全に正本状態へ復元してから後続試行を実行できます。致命的な準備／実行失敗では、そのベンチマーク呼び出しを停止し、残る未開始試行だけをskippedとします。失敗とskipは明示的に残し、性能標本にはしません。

<a id="cpu-backends-and-parallelism"></a>

## CPUバックエンドと並列性

各CPUバックエンドには、安定した名前とroleがあります。

- `production`：主要な性能比較系列。
- `reference`：正当性確認、smoke test、任意の最適化依存ライブラリがない場合の実行。

referenceバックエンドがproductionバックエンドを黙って置き換えてはいけません。実効設定の各ベンチマークエントリは、`default_speedup_cpu_backend`を指定します。speedupはその名前のproductionバックエンドに対してだけ生成します。利用できない、または無効ならspeedupを省略して理由を記録し、reference系列へフォールバックしません。

primary productionキャンペーンは、ベンチマークごとにproduction CPUバックエンドをちょうど1つ選択します。CPU/CUDA/OpenACCの3実装の順列に、複数のCPUバックエンドを含めてはいけません。追加のreferenceバックエンドや別のproductionバックエンドは、別キャンペーンか、明示的に分類したauxiliary系列として測定します。primary speedupの分母に自動採用することはありません。実効設定とnodeメタデータには、系列が`primary`か`auxiliary`かを記録します。

raw結果では次を区別します。

- `cpu_threads_requested`：CLI／設定が要求した値。
- `cpu_threads_effective`：バックエンドが実際に使うスレッド数。確定できなければnull。
- `cpu_parallelism`：`serial`、`threaded`、`unknown`。

実効設定は、各系列の`cpu_backend_role`、`series_role`、予定される並列性／既知の有効スレッドメタデータの正本です。`run_suite.py`はその値をベンチマークへ渡し、出力行を同じseriesオブジェクトと照合します。共通resultコードは、バックエンド名の部分文字列からroleを推測したり、特定バックエンド名だけを特別扱いしてseries roleを選んだり、non-serial名をすべてthreadedに分類したりしてはいけません。より良い観測値がないという理由で、要求スレッド数を`cpu_threads_effective`へコピーすることはありません。観測できない有効スレッド数はnull、確定できない並列性は`unknown`です。

バックエンドごとのスレッド制御は次のとおりです。

- 呼び出し側ソースにOpenMP並列ループがなくても、OpenMPランタイム変数がリンク先ライブラリを制御する場合があります。`OMP_NUM_THREADS`、`OMP_PROC_BIND`、`OMP_PLACES`を記録してください。アプリケーションの`-fopenmp` オプションの有無では、ライブラリのスレッド実装を特定できません。
- oneMKLは`MKL_NUM_THREADS`または対応するローカルスレッド制御APIを使い、要求値を確定した有効スレッド数と別に記録します。保存済みPegasusキャンペーンは`MKL_NUM_THREADS=48`、`MKL_THREADING_LAYER=INTEL`を使いましたが、oneMKL raw行は`cpu_threads_effective=null`のままであり、有効スレッド数48の測定値ではありません。
- OpenBLASは`OPENBLAS_NUM_THREADS`または対応バックエンド APIを使います。ビルドが有効スレッド数を確定できない場合は、要求値を実効値と主張せず、nullと`unknown`を記録します。
- threaded FFTWはスレッド並列用インターフェースを初期化し、プラン作成前に要求スレッド数を適用します。逐次FFTWの有効スレッド数は常に1です。保存済み教材掲載用ビルドは、FFTWのpthread版`--enable-threads`と、48スレッドのFFTW threads APIを使っています。cuFFTのCPU教材サンプルは逐次であり、このthreaded設定はベンチマークのものです。

FFTWの逐次／threaded系列は、`cpu-fftw-serial`と`cpu-fftw-threaded`のように別名を使います。oneMKL/OpenBLAS系列も、`cpu-onemkl`や`cpu-openblas`という実装固有の名前を使います。`cpu-reference-csr`はreferenceバックエンドです。

Pegasusの正本教材掲載用設定では、cuFFTは`cpu-fftw-threaded`だけです。別途呼び出す`cpu-fftw-serial`実行ファイルは教材との対応確認用に利用できますが、教材掲載用系列ではなく、threadedバックエンドの代替にはなりません。

正本のcuRAND CPUベンチマークは、`std::mt19937_64`と`std::uniform_real_distribution<double>`を使う`cpu-std-random-serial`で、roleは`production`、有効スレッド数は1です。正本のThrust CPUベンチマークは、execution policyなしの`std::transform_reduce`を使う`cpu-stl-serial`で、roleは`production`、有効スレッド数も1です。承認済み教材掲載用凡例は両方を明示的に**single thread**とし、並列、同一アルゴリズム、48-core CPUの性能とは呼びません。正確な表示ラベルは[教材掲載用図](#publication-figures)を参照してください。

<a id="fortran-binding-of-this-protocol"></a>

## この規約とFortran実装の対応

`nvidia/fortran`でも同じCLI、数学、FP32/FP64精度、検証閾値、試行前復元、compute・one-shot E2Eの契約を使います。Fortran workloadから`ISO_C_BINDING`で既存Cタイマー・CLI・結果出力へ接続し、計算ライブラリ呼出しはFortranに残します。1始まりCSRは同じPoisson問題の表現です。CPU solver benchmarkは`dgetrf`＋`dgetrs`で、教材の`dgesv`とは分けます。

Fortran CPU production backendは`cpu-fftw-serial`/`cpu-fftw-threaded`、`cpu-onemkl`、`cpu-fortran-random-serial`、`cpu-fortran-sum-serial`です。後2者は組込み関数を使い、自動並列化・GPU offloadなし、実効1とコンパイラ識別情報を記録し、C++ MT19937/STLと名乗りません。FFTWはplan前にthreads APIを設定し、oneMKLはAPIで要求数を設定しても未観測の実効数はnullです。CPU乱数は`cpu_engine="Fortran random_number"`を記録し、seed配列を設定seed（LP64符号付き範囲）で埋め、offset分を消費します。GPUはC/C++版と同じcuRAND generator契約で、CPU/GPUの数列一致は要求しません。

Fortran Release既定値は`-O3`です。GNU targetは`-fno-fast-math -ffp-contract=off`、NVHPC targetは`-Kieee -Mnoflushz -Mnodaz`を使い、GPU targetだけにSeparate Memory ModeやOpenACCを明示します。このprofileでは未承認global数値flags、default kind変更、自動並列化/offload、unsafeなbridge flagsを拒否します。`NVCOMPILER_FPU_STATE`はunsetし、NVHPC benchmarkでは非空設定を拒否します。過去のC/C++の`-fast`条件を変更したり遡って承認したりはしません。verificationは明示的で、Releaseでも要求どおり実行します。

Fortran診断configは`configs/fortran/pilot.json`で、本測定profileの承認やC/C++ publication workloadとは別です。言語識別と互換性は[`RESULT_SCHEMA.md`（英語）](RESULT_SCHEMA.md)が定めます。campaignを分け、既存2パネル図とFortran固有ラベルを使い、C/C++測定値を流用しません。

<a id="timing-fields"></a>

## 時間測定フィールド

各raw試行について、次のとおり定義します。

```text
elapsed_total_sec = total time actually measured in the trial
elapsed_sec       = elapsed_total_sec / repeat
```

どちらも単位は秒で、成功試行では有限かつ0以上でなければなりません。結果スキーマは以下を別々に記録します。

- `record_timestamp`：raw行を作成した時刻。
- `measurement_start_timestamp`：試行の最初の測定区間のUTC開始時刻。
- `measurement_end_timestamp`：試行の最後の測定区間のUTC終了時刻。

end-to-endスコープでは、個別に測るrepeatの間に、非計測の状態復元と後処理が入ることがあります。このため、測定開始から測定終了までのUTC上の時間幅は、測定区間だけの和である`elapsed_total_sec`と一致する必要はありません。

<a id="compute-scope"></a>

### computeスコープ

GPUの一般的な順序は次のとおりです。

1. メモリ確保、転送、プラン、ハンドル、記述子、作業領域を準備する。
2. 正本の試行開始状態を復元する。
3. 測定前に同期する。
4. 開始時刻を読む。
5. 演算を`repeat`回実行する。
6. 終了時刻の前に同期する。
7. 終了時刻を読む。

メモリ確保、ホスト-デバイス転送、プラン/ハンドル/記述子/作業領域作成、入力復元、検証、後処理は測定区間外です。CPU実装は同じタイマーと対応する演算境界を使います。

cuFFTプラン、cuBLASハンドル、cuSPARSE記述子/作業領域、cuRAND生成器は測定前に作成します。破壊された入力の復元も測定区間外です。cuSOLVER computeスコープは`repeat = 1`を要求し、より大きい値はエラーです。代わりに複数試行を使ってください。

<a id="end-to-end-scope"></a>

### end-to-endスコープ

ホスト入力のメモリ確保と初期値生成は測定区間外です。各repeatを個別に測ります。

1. 開始時刻を読む。
2. ライブラリ固有の準備、デバイスメモリ確保、H2D、演算、必要な同期、D2H／結果取得を行う。
3. 終了時刻を読む。
4. そのrepeatの後処理を終了時刻の後に行う。

後処理、検証、出力は測定区間外です。ホスト入力の再生成も区間外です。個別に測ったrepeat時間を合計して`elapsed_total_sec`とします。

<a id="openacc-compute-scope"></a>

### OpenACCのcomputeスコープ

OpenACCで管理するメモリ確保と移動は`PROJECT_SPECIFICATION.md`が定めます。準備順序は固定です。

1. 管理配列のデバイスアドレスを必要としないプラン/ハンドルを作成する。
2. `acc data` 領域へ入り、OpenACCランタイムが管理配列を確保して、指定したcopyin/createを行う。
3. `host_data use_device`で、すでにデバイス上にある管理データをデバイスアドレスへ変換する。
4. そのデバイスアドレスを必要とする記述子を作成する。
5. 作業領域 sizeを照会し、OpenACCデータ clauseで管理しないライブラリ作業領域だけを明示的に確保する。
6. 同期する。
7. 開始時刻を読む。

ライブラリが使わない手順は省略できますが、該当する手順の順序は変えてはいけません。開始時刻の後、CUDAライブラリ呼び出しを`repeat`回実行し、終了時刻の前に同期してから終了時刻を読みます。OpenACCデータ exit、copyout、検証、記述子/プランの破棄、後処理はその後に行い、測定区間外とします。

cuSOLVERのように、永続的なデータ記述子ではなくrawアドレスを必要とする呼び出しでは、アドレス変換より前にデータ領域へ入ります。作業領域確保前にはバッファ-size照会に必要な最小限の変換、演算中には`getrf`/`getrs`に必要な最小限の変換を行います。これは、アドレス変換前にデータがデバイス上に存在しなければならないという規則を変えるものではありません。

<a id="openacc-end-to-end-scope"></a>

### OpenACCのend-to-endスコープ

各repeatで、プラン/ハンドル作成前に開始時刻を読みます。測定パイプラインには次を含めます。

- プラン/ハンドル作成。
- OpenACCデータ領域への進入。ランタイムのメモリ確保と指定したcopyin/createを含む。
- デバイス-アドレス変換と、ポインタに依存する記述子作成。
- ライブラリ作業領域のsize照会と確保。
- CUDAライブラリ呼び出し。
- `cudaDeviceSynchronize`。
- 必要なcopyoutを含むデータ-領域からの退出。
- ホストで結果が利用可能になるまでの処理。

データ exit/copyout完了後にだけ終了時刻を読みます。プラン、ハンドル、記述子、明示的作業領域は、その後に非計測の後処理として破棄します。OpenACC end-to-end測定では、データ領域への出入りやcopyin/copyoutを省略してはいけません。

<a id="warm-up-and-canonical-trial-start-state"></a>

## warm-upと正本の試行開始状態

各実装・問題サイズについて、最初に準備、warm-up、warm-upで変更したすべての入力・出力・ライブラリ状態の完全な復元を行います。その後、各raw試行に独立に次の規則を適用します。

```text
restore canonical trial-start state
run one measured trial
```

warm-up後の復元を行った直後でも、最初の試行を再び復元します。いずれの試行も、前試行の最終状態を引き継いではいけません。

warm-upは、測定に選んだスコープ固有のパイプラインを実行します。computeスコープでは、永続的なプラン、ハンドル、記述子、生成器、デバイスメモリ確保、作業領域を先に作り、compute演算をちょうど`warmup`回実行し、正本状態を復元してから、同じ永続コンテキストで試行を測定します。end-to-endスコープでは、compute専用の永続コンテキストを作成・保持しません。代わりに、スコープのメモリ確保、準備、転送、演算、結果取得、完全な後処理を含む、一時的な完全パイプラインをちょうど`warmup`回実行します。最初のend-to-end試行の開始時に、warm-upのハンドル、記述子、生成器、作業領域、デバイスメモリ確保、デバイスコンテナを残してはいけません。

computeスコープでは、1試行のrepeat loop内でCやyを順次更新できます。検証は全`repeat`回の更新を反映しますが、次のraw試行前にC/yをresetします。end-to-endスコープでは、各repeatの測定区間外でホスト側の入力・出力を復元し、すべてのパイプラインが同じ正本初期値から始まるようにします。

必須の復元処理は次のとおりです。

- cuBLAS：各試行前と、各end-to-end repeat前にCを復元する。
- cuSPARSE：各試行前と、各end-to-end repeat前にyを復元する。
- cuSOLVER：各試行前にA、B、`ipiv`、`getrf_info`、`getrs_info`を復元する。`repeat`は必ず1のままとする。
- cuRAND compute：各試行前にシード、オフセット、生成器状態を復元する。その試行内のrepeatは、その状態から連続するブロックを生成する。
- cuRAND end-to-end：各測定repeat内で生成器を作成しシード/オフセットを設定して、すべてのrepeatに同じ開始条件を与える。
- 将来のin-place cuFFT／Thrust演算：次の試行またはend-to-end repeat前に、破壊されたすべての入力を復元する。

実装ごとのwarm-upは、その実装の測定直前に行います。必須の試行前・repeat前の復元はすべて非計測です。補助処理が時間測定を汚染せずに復元状態を検証できる場合は、検証すべきです。

<a id="standard-workloads-and-verification"></a>

## 標準ワークロードと検証

ベンチマークの数学的定義とサンプルの既定入力は`PROJECT_SPECIFICATION.md`に従います。以下のスイープと検証規則はベンチマーク固有です。

<a id="verification-schema-and-thresholds"></a>

### 検証スキーマと閾値

すべての閾値は実効設定に置きます。説明のない許容誤差のmagic numberをソースに含めてはいけません。通常の数値誤差は次を満たすと成功です。

```text
error <= abs_tolerance + rel_tolerance * reference_scale
```

スイート実行では、実効設定に各ベンチマークの完全な検証オブジェクトを要求し、すべての値をベンチマークのコマンドラインへ明示的に渡します。ランナーは値の欠落を拒否します。単独呼び出しでは、文書化したsmoke既定値を維持します。絶対許容誤差は`1e-12`、相対許容誤差は`1e-10`ですが、cuFFTは絶対`1e-4`、相対`1e-5`です。cuRANDは`sigma_multiplier=6`、`expected_mean=0.5`、`expected_second_central_moment=1/12`を使います。これらの組込み既定値はproduction-スイート設定ではなく、実効設定を上書きしません。

単一指標の場合も含め、すべてのベンチマークが次のオブジェクトを出力します。

- `verification_metrics`：指標名から測定値への対応。
- `verification_thresholds`：指標名から完全な条件への対応。手法と、必要なreference scale、絶対／相対許容誤差、範囲境界を含む。
- `verification_primary_metric`：代表指標名またはnull。
- `verification_status`：`pass`、`failure`、`skipped`、`nonfinite`。

metricsとthresholdsにscalar形式はありません。nonfiniteな指標はnullで表し、`verification_status = nonfinite`、全体の`status = failure`、診断メッセージを記録します。検証失敗はraw結果に保持しますが、性能集計からは除外します。

検証は、raw出力、中間の誤差／残差、ノルムのすべてについて、`fmax`、`std::max`などのreductionへ渡す前に有限性を確認します。`fmax`も`std::max`もNaN検出器ではありません。閾値内の有限値は成功、閾値外の有限値は検証失敗、NaNまたは正負いずれかの無限大は常にnonfinite検証失敗です。JSON検証オブジェクトの確保や必須指標/閾値の挿入に失敗した場合は、致命的なresult-construction／ベンチマークエラーであり、数値検証の失敗でも、部分的な成功レコードでもありません。

<a id="cufft-1"></a>

### cuFFT

- batched 1D C2C forward FFT、FP32を、主としてFFT長Nについてスイープする。
- 1つのスイープ内でbatch、transform、精度を固定する。
- CPUではFFTW3の単精度interfaceを使う。threaded FFTWは別名の任意productionバックエンドとする。
- 期待するDC成分と最大non-DC誤差を、`dc_relative_error`、`non_dc_max_abs_error`などの独立した指標で記録する。

<a id="cublas-1"></a>

### cuBLAS

- FP64 DGEMMを、通常`m = n = k = N`としてスイープする。
- 名前を持つCBLAS互換productionバックエンドを使い、oneMKL、OpenBLAS、その他のどの実装かを記録する。
- 検証には全要素1の期待値を使い、`repeat`によるすべてのC更新を反映する。

<a id="cusparse-1"></a>

### cuSPARSE

- 正本のFP64 Poisson CSR SpMVを、通常`nx = ny`として、主に未知数総数`N = nx * ny`についてスイープする。
- oneMKL Sparse BLASなど、名前を持つ最適化productionバックエンドを優先する。`cpu-reference-csr`は独立したreference系列とする。
- 行オフセット、列インデックス、値、数値結果を検証する。
- 検証では`repeat`によるすべてのy更新を反映する。

<a id="cusolver-1"></a>

### cuSOLVER

- 密FP64 LU因子分解と求解を、主に行列次数Nについてスイープし、スイープ内で`nrhs`を固定する。
- CPUベンチマークは`LAPACKE_dgetrf`の後に`LAPACKE_dgetrs`を使い、CPU教材サンプルの`LAPACKE_dgesv`呼び出しは使わない。
- CUDA/OpenACCベンチマークは`getrf`の後に`getrs`を使う。
- `getrf_info`と`getrs_info`を別々に保存する。
- `solution_relative_error`と`relative_residual`を独立した検証指標として保存する。
- `repeat = 1`を要求し、複数試行を使う。

<a id="curand-1"></a>

### cuRAND

- 一様倍精度乱数生成を、主として要素数Nについてスイープする。
- 実際のCPUエンジン、cuRAND生成器、シード、オフセット、orderをメタデータに記録する。
- CPU/GPUストリーム間の要素ごとの一致は要求しない。
- `std::mt19937_64`と`CURAND_RNG_PSEUDO_DEFAULT`は異なる乱数アルゴリズムである。経過時間が示すのは同じタスク、出力型、一様分布の比較であり、同一アルゴリズムによるCPU/GPU比較ではない。
- CPU/GPUの範囲健全性確認はどちらも`0.0 <= x <= 1.0`を許容し、各バックエンドの正確な区間契約を`verification_thresholds`または`parameters`に記録する。
- 共通の端点を含む健全性範囲を適用しつつ、CPU標準分布の契約を`[0,1)`、cuRANDの契約を`(0,1]`と記録する。
- 設定は`sigma_multiplier`、`expected_mean`、`expected_second_central_moment`を保持し、初期値はそれぞれ`6.0`、`0.5`、`1/12`とする。
- 必須指標は`observed_min`、`observed_max`、`sample_mean`、`second_central_moment_about_half`であり、次のとおり定義する。

  ```text
  second_central_moment_about_half = mean((x_i - 0.5)^2)
  ```

- 平均の検証は次のとおり。

  ```text
  abs(sample_mean - 0.5)
      <= sigma_multiplier * sqrt(1 / (12 * N))
  ```

- 第2中心モーメントの検証は次のとおり。

  ```text
  abs(second_central_moment_about_half - 1/12)
      <= sigma_multiplier * sqrt(1 / (180 * N))
  ```

- 検証は最後に取得したN個の値を使い、`verification_sample_count=N`を記録する。
- 図、plotメタデータ、aggregateメタデータ、利用者向け文書には、**同じ分布・出力型のタスクを比較するが、RNGアルゴリズムは異なる**ことを明示する（原文の表示文言：**Same distribution and output type task; different RNG algorithms.**）。

<a id="thrust-1"></a>

### Thrust

- FP64の`transform_reduce`を、主として要素数Nについてスイープする。
- CPUバックエンド名と有効な並列性を記録する。
- 設定した数値閾値を使い、期待値Nと比較して検証する。

<a id="interleaved-execution-order"></a>

## 実装を交互に実行する順序

CPUの全ケース、CUDAの全ケース、OpenACCの全ケースという順の実行は禁止します。各ノードでは次の順序です。

```text
for each benchmark:
    determine this node's size order
    for each problem size in that order:
        run CPU, CUDA, and OpenACC in the assigned implementation permutation
```

同じ問題サイズの3実装は、時間的に近接させて実行する必要があります。各実装のwarm-upは、その測定直前に行います。

実装の順列は次のとおりです。

| インデックス | 順序 |
| --- | --- |
| 0 | CPU, CUDA, OpenACC |
| 1 | CPU, OpenACC, CUDA |
| 2 | CUDA, CPU, OpenACC |
| 3 | CUDA, OpenACC, CPU |
| 4 | OpenACC, CPU, CUDA |
| 5 | OpenACC, CUDA, CPU |

次を使います。

```text
permutation_index = (node_index + wave) mod 6
size_order_index  = node_index mod 2
```

`size_order_index = 0`は設定の順序、インデックス 1はその逆順を使います。実装順とサイズ順は、意図的に同じ式を使いません。

6ノードなら、連続する2 waveによって、すべての実装順列に両方のサイズ順が割り当たります。5ノードと8ノードのwaveも許可しますが、メタデータに割当数を記録して不均衡を明示しなければなりません。nodeメタデータには`implementation_order`、`permutation_index`、ベンチマークごとの実際の`size_order`、`size_order_index`を記録します。

<a id="blocks-waves-and-aggregation"></a>

## ブロック・waveと集計

`node_index`は現在のjobにおけるMPI rankであり、固定の物理ノード識別子ではありません。物理ノードは`hostname`で識別します。ブロックは、`RESULT_SCHEMA.md`で定義するrun ID、wave、hostnameの組合せです。

集計は階層的に行います。

1. ブロック階層で、ベンチマーク／実装／問題／スコープごとに、有効試行の中央値を計算する。
2. 同じブロック内で、設定されたproduction CPU中央値を使い、CPU/CUDAとCPU/OpenACCのspeedupを計算する。
3. wave階層で、hostname間のブロック中央値を集計し、wave中央値とその他のwave内統計を生成する。
4. primary wave間階層では、各waveのwave中央値だけを1つの入力値として使い、`summary_input_statistic = "wave_median"`とする。

四分位数の四分位数、wave IQRの平均、全ブロックの再poolをprimary wave間要約にしてはいけません。primary wave間の各統計量は、wave中央値のベクトルから計算し直します。

任意のpooled解析は、`pooled exploratory summary`とラベル付けした独立レコードであり、primary結果やprimary図の入力ではありません。全CPU値の中央値を全GPU値の中央値で割る方法でspeedupを計算してはいけません。

有効標本が2個以上ある各groupには、Python 3.9標準ライブラリの次の定義を使います。

```python
statistics.quantiles(values, n=4, method="inclusive")
```

中央値、Q1、Q3、`IQR = Q3 - Q1`、最小値、最大値を記録します。有効標本が1個の場合は次のとおりです。

- 中央値、最小値、最大値は、その値とする。
- Q1、Q3、IQRはnullとする。
- `aggregate_status = "insufficient_sample_count"`とする。

有効標本が0個の場合は数値要約を出力せず、`aggregate_status = "no_valid_samples"`の状態のみの aggregateレコードを出力します。

primary production測定には最低5ノード、望ましくは8ノードを使います。追加waveは別の時間帯を対象にできます。thermal、power、clock、テレメトリ、検証の異常をフラグ付けしますが、経過時間だけを理由に遅いノードを除去してはいけません。割当数、実際の失敗による除外、有効標本数はaggregateメタデータに明示します。

<a id="publication-figures"></a>

## 教材掲載用図

教材掲載用描画は、primaryの`cross-wave` 要約レコードだけを使います。有効な教材掲載用ライブラリごとに、2パネルの実行時間図を1枚出力します。

- `cufft-elapsed-time.png`
- `cublas-elapsed-time.png`
- `cusparse-elapsed-time.png`
- `cusolver-elapsed-time.png`
- `curand-elapsed-time.png`
- `thrust-elapsed-time.png`

左パネルのタイトルは**Library kernel execution time**で、データ常駐computeを表します。右パネルは**End-to-end execution time**で、ホスト入力からホスト出力までを1回で処理するパイプラインを表します。両方ともx軸にライブラリ固有の問題サイズ、y軸に**Elapsed time [ms]**を用い、値が小さいほど良好です。CPU、CUDA、OpenACCを同じ図に表示します。computeのプロット値は、既存の1演算あたり`elapsed_sec`を秒からミリ秒へ変換した値であり、repeatで再度割りません。1回の値は、`repeat=1`のend-to-endレコードから取得します。speedup、throughput、bandwidth、FLOPS、sample-rate、element-rate、reuse-count、amortized、break-even、その他の汎用elapsed図は生成しません。

承認済みの共通凡例は2パネルの下に配置します。正確なラベルは次のとおりです。

| 系列 | ラベル |
| --- | --- |
| cuFFT CPU | `Intel Xeon Platinum 8468, FFTW (48 C)` |
| cuBLAS/cuSPARSE/cuSOLVER CPU | `Intel Xeon Platinum 8468, oneMKL (48 C)` |
| cuRAND CPU | `Intel Xeon Platinum 8468, std::mt19937_64 (single thread)` |
| Thrust CPU | `Intel Xeon Platinum 8468, STL (single thread)` |
| CUDA | `NVIDIA H100 PCIe, CUDA` |
| OpenACC | `NVIDIA H100 PCIe, OpenACC` |

`(48 C)`のラベルは承認済みキャンペーン設定を説明するもので、前述のoneMKL有効スレッド数の根拠を変更しません。cuRAND/Thrustは逐次referenceであり、並列または同一アルゴリズムの比較分母ではありません。対数x軸は2進K/M表記（`1K=1024`、`1M=1048576`）を使います。cuSOLVERは該当入力で`4K`、`8K`、`12K`を維持し、別のスイープでは実際の入力サイズを使います。この表は承認済み教材設定であり、不明なハードウェアの既定値ではありません。再現には、明示的な`compact-requested`規約の`--display-config`を使ってください。通常のラベルは要求数と報告された有効数を区別し、欠けた値はunknownのままです。`--node-metadata`で観測CPU識別情報を、raw行からGPU識別情報を取得できます。provenanceと検証は[表示スキーマ（英語）](RESULT_SCHEMA.md#plot-display-configuration)が定めます。[自己測定の手順](PORTABILITY.ja.md#figures-from-your-own-measurements)に従ってください。

Thrust図には、成功したCUDA/OpenACC行が同一の`library_version`を報告するraw-result根拠が必要です。根拠の欠落、混在、不一致は、いずれの図も書き出す前にエラーとします。1実装を黙って省略してはいけません。

各教材掲載用キャプションは、plotのrun IDと実行環境ハッシュを、既存の不変run/nodeメタデータへ対応付けます。結果スキーマフィールドは追加しません。システムラベル、CPUモデル、GPUモデル、精度、演算、6ノード×2 wave、nodeブロックごとに5試行、ブロック中央値→wave中央値→wave間中央値という集計、両方の測定境界、測定増幅としてのcompute repeat、end-to-end repeat 1、値が小さいほど良いことを記載します。さらに次も記録します。

- cuBLAS：cuBLAS default mathモードによるFP64 DGEMM。
- cuFFT：FP32 complex batched 1-D C2C forward変換、`batch=4096`、2の冪の長さ。
- cuSPARSE：規則的な2-D Poisson行列のFP64 CSR SpMV。追加preprocess段階なし。
- cuSOLVER：`nrhs=16`のFP64 LU因子分解と求解。
- cuRAND：GPU pseudo-default生成器と、同一アルゴリズムではない逐次CPU生成器による一様倍精度乱数生成。
- Thrust：CUDA/OpenACCに共通する1つのCCCL/Thrustバージョンによるdouble `transform_reduce`。

2パネルは教材上のモデルでもあります。データ常駐computeは、ライブラリ演算のためにデータをデバイスへ保持するアプリケーションを表します。1回のパイプラインは準備、デバイスメモリ確保、H2D、1演算、完了、D2Hを、ホストで結果が利用可能になるまで含みます。入力生成、検証、シリアライズ、ファイルI/O、プロトコル上の後処理は区間外です。デバイス常駐データを再利用すると、転送の相対的寄与が小さくなり、観測されるアプリケーションのコストが1回の側からcompute側へ近づき得ます。これは既存の2スコープの解釈であり、第3のamortizedスコープではありません。ワークロードはライブラリごとに1つの代表的演算であり、そのライブラリの全アルゴリズムを特徴付けるものではありません。

<a id="reading-the-figures-and-applying-the-results"></a>

## 図の読み方と結果の適用

図は2つの境界の比較であり、アプリケーション全体のspeedupを保証しません。x軸は設定した問題サイズ、y軸は1演算あたりのミリ秒と読み、値が小さいほど良好です。同じサイズ、同じパネル内でだけCPU、CUDA、OpenACCを比較してください。computeのプロット値を`repeat`で再度割ってはいけません。

| 処理 | computeパネル | 1回の E2Eパネル |
| --- | --- | --- |
| ホスト入力の確保／生成、正本状態の復元 | 区間外 | 区間外 |
| GPUメモリ確保、H2D、プラン/ハンドル/記述子/作業領域の準備 | 測定前に準備 | ライブラリが必要とする場合、各repeatのパイプライン内 |
| ライブラリ演算と、その完了を確定する同期 | 区間内 | 区間内 |
| D2H／結果取得 | compute測定後 | 終了時刻前の区間内 |
| OpenACCデータ領域への出入りと必要なcopyin/copyout | 区間外 | 区間内 |
| 結果取得後の明示的後処理、数値検証、シリアライズ、ファイルI/O | 区間外 | 区間外 |

CPU実装は対応する演算／準備境界を使い、存在しないホスト-デバイスコピーを仮定しません。ライブラリ固有のリソース詳細、とりわけcuSOLVERの状態復元とFFTプランについては、上のスコープ/ワークロード節と各ライブラリREADMEに記載しています。E2Eはプログラム全体の実経過時間**ではありません**。入力生成、検証、出力、結果取得後の後処理は除外します。

warm-upは選択スコープに従い、非計測です。各raw試行は復元済み状態から始まります。computeの反復は測定可能な区間を増幅します。教材掲載用のE2E repeatは1、cuSOLVER repeatは常に1です。有効試行からブロック中央値、ブロックからwave中央値、wave中央値からwave間中央値を求めます。1ブロック/waveはパイプライン確認に有用ですが、ノード間／時間帯の変動を確定するものではありません。プロットした中央値だけでなく、標本数、四分位数、失敗数、provenanceも確認してください。

結果を自分のプログラムへ適用する前に、データがGPUへ常駐するか、プラン/ハンドルを再利用できるか、転送・同期がどの頻度で必要か、この演算がアプリケーション時間のどれだけを占めるかを確認してください。GPU computeが短くても、1回のパイプラインは遅いことがあります。その差は調査すべきコストを示唆しますが、独立に測定した転送だけの時間でも、特定ボトルネックの証明でもありません。ホスト処理、I/O、他kernel、競合が全体の改善を制限し得ます。overlapや再利用の最適化は、実装可能というだけで測定済みにはなりません。

runを比較するときは、精度、演算、パラメータ、検証閾値、CPUプロバイダ、要求／有効スレッド数の根拠、コンパイラフラグ、GPU/ソフトウェアバージョン、測定スコープを明示してください。設定した全要素1または解析的な問題は制御された例であり、実世界の入力分布の調査ではありません。cuRANDは異なるRNGアルゴリズムを比較し、cuRAND/ThrustのCPU系列は単一スレッドです。その比を、最適化された並列CPU実装に一般化してはいけません。failed、skipped、nonfinite試行はrawデータに残して数値集計から除外し、都合の悪いデータとして消去しません。

設定の正確な編集箇所と実行可能なコマンド列は、[読者向けワークフロー](PORTABILITY.ja.md#measuring-on-your-own-system)を参照してください。
