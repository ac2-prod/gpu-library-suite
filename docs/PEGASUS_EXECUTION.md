# Pegasus実行仕様

## 文書の所有範囲

この文書は、Pegasus固有のresource、PBS/NQSV、module、OpenMPI、scratch、
telemetry、build/run手順だけを所有する。portableなCLIと測定規約は
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md)、結果・metadata形式は
[`RESULT_SCHEMA.md`](RESULT_SCHEMA.md)、数学的問題は
[`PROJECT_SPECIFICATION.md`](PROJECT_SPECIFICATION.md)を正本とする。

ここで記載するhardwareやscheduler情報をportable sourceへ埋め込まない。
個人名、個人用絶対path、特定login hostは保存せず、共有pathは
`/work/<project>/<user>/...`のようなplaceholderで表す。

## Pegasus profile

- scheduler: NEC NQSV
- 最大150ノード
- 1ノード: Intel Xeon Platinum 8468、48物理core、Hyper-Threading無効
- 1ノード: NVIDIA H100 80 GBを1枚
- 1ノードの実効memory: 約115 GB
- node-local filesystem: `/scr`
- node-local persistent-memory領域: `/pmem`
- shared filesystem: `/work`
- 1ノードに同時に1jobを割り当てるnode-exclusive方式
- OS: Ubuntu 22.04

初期版では`/pmem`を使用しない。portable sourceへH100、GPU architecture、
Pegasus固有path、module、queue、scheduler directiveを記載しない。

## Queue profileと人間による確認

| Queue | 制限・用途 |
| --- | --- |
| `debug` | `qlogin`によるinteractive verification。最大1時間、最大2ノード、0 point |
| `gpu` | `qsub`によるbatch smoke、pilot、benchmark。最大24時間、最大150ノード、projectによって利用可否が異なる |
| `gen_S` | 一般利用/HPCI向け、1〜31ノード、最大24時間。AC2では使用しない |
| `gen_M` | 一般利用/HPCI向け、32〜63ノード、最大24時間。AC2では使用しない |
| `gen_L` | 一般利用/HPCI向け、64〜150ノード、最大24時間。AC2では使用しない |

利用可能queueはprojectによって異なる。account、queue、module名・version、
MPI version、shared output pathを推測または固定しない。投入前に人間が次を
実行して確認する。

```text
qstat -Q
pegasusinfo
rbudgetcheck
module avail
module list
```

Codex、renderer、build script、runnerはこれらを自動実行しない。job投入、
削除、照会も人間だけが行う。

## Pegasus設定

`jobs/pegasus/pegasus.json.example`はPython 3.9標準libraryで読める
JSONとし、少なくとも次を明示設定させる。

- account
- queue
- node数
- walltime
- MPI version
- CPU thread要求
- `cpu_cuda_build_modules`
- `openacc_build_modules`
- `benchmark_runtime_modules`
- module purgeの有無
- compiler/build environment
- CUDA Toolkit root/version
- `NVHPC_CUDA_HOME`、またはNVHPCが実際に選択するCUDA Toolkit
- shared result root
- `LOCAL_SCRATCH_ROOT`（標準は`/scr`）
- OpenMPI追加option
- optional environment overrides
- `OMP_NUM_THREADS`
- `OMP_PROC_BIND`
- `OMP_PLACES`
- `OMP_DYNAMIC=FALSE`
- `MKL_NUM_THREADS`
- `MKL_DYNAMIC=FALSE`
- `MKL_THREADING_LAYER`
- `OPENBLAS_NUM_THREADS`

account、queue、MPI version、`benchmark_runtime_modules`など必須値が未設定なら
rendererは失敗する。`module purge`は設定で明示された場合だけ実行する。

`cpu_cuda_build_modules`はCPU/CUDA build時だけ、
`openacc_build_modules`はOpenACC build時だけ使用する。benchmark jobは両build
module群を無条件に同時loadしない。人間が`benchmark_runtime_modules`として、
prebuilt CPU/CUDA/OpenACC binaryを同一job内で実行できる互換runtime環境を
指定する。

build profileは次の2つだけとする。

```text
cpu-cuda:
  GPU_SUITE_BUILD_CPU=ON
  GPU_SUITE_BUILD_CUDA=ON
  GPU_SUITE_BUILD_OPENACC=OFF

openacc:
  CMAKE_C_COMPILER=nvc
  CMAKE_CXX_COMPILER=nvc++
  GPU_SUITE_BUILD_CPU=OFF
  GPU_SUITE_BUILD_CUDA=OFF
  GPU_SUITE_BUILD_OPENACC=ON
```

OpenACC buildは、設定された`NVHPC_CUDA_HOME`またはnvc++が一意に報告した
CUDA Toolkitを`CUDAToolkit_ROOT`として使用する。CMakeが解決したToolkitと
NVHPCが選択したToolkitのpath/versionが一致しなければbuildを失敗させる。

OpenACCのcuFFT、cuBLAS、cuSPARSE、cuSOLVER、cuRAND targetは
`OpenACC::OpenACC_CXX`、`CUDA::cudart`、対応する`CUDA::<library>` imported
targetで統一してlinkする。`-cudalib=<library>`方式とは混在させない。
OpenACC Thrustは`OpenACC::OpenACC_CXX`と`CUDA::cudart`をlinkし、compile/link
双方へNVHPCの`-cuda`を付ける。各CUDA-library targetは同じlink設定で、header、
`cudaGetDeviceCount`、実際に利用するlibrary symbolを含むnvc++
compile-and-link probeを通過しなければならない。

## PBS template

PBS templateは1行目を必ず`#!/bin/bash`とし、`#!/bin/sh`を使用しない。
基本形は次とする。

```bash
#!/bin/bash
#PBS -A @ACCOUNT@
#PBS -q @QUEUE@
#PBS -b @NODES@
#PBS -l elapstim_req=@WALLTIME@
#PBS -N @JOB_NAME@
#PBS -o @PBS_STDOUT@
#PBS -e @PBS_STDERR@
#PBS -T openmpi
#PBS -v NQSV_MPI_VER=@MPI_VERSION@
#PBS -v OMP_NUM_THREADS=@CPU_THREADS@
```

`-A`と`-q`は必須とする。processes per node × threads per processは48以下。
初期版は1ノード1processであり、標準CPU thread要求は48とする。要求値は
JSON設定から変更できる。

標準出力と標準errorはNQSVの一時spoolへ大量に残さず、設定された
`/work/<project>/<user>/...`配下の明示pathへ保存する。

## ModuleとOpenMPI起動

job本体は次の順で準備する。

1. 設定で要求された場合だけ`module purge`を行う。
2. 設定された`benchmark_runtime_modules`だけをloadする。
3. `module load openmpi/$NQSV_MPI_VER`を行う。
4. `module list`をlogへ保存する。
5. `cd "${PBS_O_WORKDIR}"`を行う。
6. `PBS_NODEFILE`をrunのlogへ保存する。
7. runtime software environmentを取得する。
8. preflightを実行し、成功後だけ1ノード1processでnode runnerを起動する。

基本起動形は次とする。

```bash
# shellcheck disable=SC2086
mpirun ${NQSV_MPIOPTS} "${MPI_OPTIONS[@]}" \
  -np @NODES@ -npernode 1 --bind-to none jobs/pegasus/run_node.sh ...
```

preflightとmeasurementの両方が同じ基本形を使う。`NQSV_MPIOPTS`はsiteが複数の
shell wordを供給するためquoteして1引数にまとめない。人間が`qsub`を行う前に、
rendered PBSの`#PBS -o`と`#PBS -e`が指す親directoryを作成し、書込み可能である
ことを確認する。rendererは親directoryの存在を検査するが作成しない。

`node_index`には`OMPI_COMM_WORLD_RANK`を用いる。1process・48threadでは
`--bind-to none`を用いる。追加OpenMPI optionは設定から与える。
Intel MPIは初期対象外。`UCX_MEMTYPE_CACHE=n`は標準強制せず、問題調査時に
設定可能とする。

## Runtime software environment取得

job masterは`benchmark_runtime_modules`をloadした後、preflight前に
`runtime-environment.json`の材料を取得する。少なくともmodule list、`PATH`、
`LD_LIBRARY_PATH`、NVIDIA driver、CUDA runtime/Toolkitのversionとpath、NVHPC
compiler/runtime、`NVHPC_CUDA_HOME`またはNVHPCが実際に選択したCUDA Toolkit、
FFTW/oneMKL/OpenBLAS/LAPACKE等のversion、全binaryの`ldd`出力、解決された
shared-library path、およびCPU thread環境の8変数を保存する。

Pegasus設定の`cuda_toolkit_version`はmodule release名ではなく、CMakeの
`CUDAToolkit_VERSION`がbuild metadataへ記録する完全なcomponent versionである。
runtime collectorはこの値をbuild metadataと完全一致で比較する。module名は
`module_list`へ別に保存し、CUDA Runtime versionは`cudaRuntimeGetVersion`から
独立に取得する。Runtime probeが失敗した場合にToolkit versionで代用しない。

benchmark runtimeの必須条件は、prebuilt binaryが必要とするdriver、CUDA
runtime、およびshared libraryが解決・実行可能であることである。`nvcc`、`nvc`、
`nvc++` commandの存在はmetadataとして取得できれば保存し、存在しなければwarning
とnull/診断を保存するが、標準runtime preflightを失敗させない。compiler commandを
runtimeにも必須とするsite固有方針は、標準条件と混同せず明示optionで有効化する。
`pegasus.json`の`require_runtime_compilers=false`が標準であり、site規則が明示的に
要求する場合だけtrueにしてrendererから`--require-runtime-compilers`を渡す。

GPU identityは各node process自身がname、UUID、NVIDIA package driver versionを
収集する。さらにcompilerを必要とせずnode-local `libcudart`をloadし、
`cudaDriverGetVersion`と`cudaRuntimeGetVersion`を呼ぶ。job masterや別nodeで得た
identityを全nodeへ流用しない。各nodeはその値を`node-metadata.json`に保存し、
benchmarkも独立してCUDA APIを呼ぶ。成功raw rowのname、UUID、Driver API version、
Runtime versionをnode metadataと一致させる。取得不能な値はnullと診断にし、
推測値で補わない。

同文書はproject-defined deterministic JSON serialization profileで保存・hashし、
`runtime_environment_sha256`を追加waveの一致条件とする。異なるruntime環境を
同じprimary cross-wave aggregateへ混在させない。同一run IDへ追加できず、新しい
run IDを要求する。

## PBS job masterによるcampaign preflight

campaign/wave metadataのwriterは、本測定用`mpirun`の外側で動作するPBS job
masterだけである。MPI-linked coordinatorや`mpi4py`を必要としない。

各waveの開始手順は次とする。

1. PBS job masterが本測定用`mpirun`より前に、同じnode allocationを使う短い
   preflight用`mpirun`を実行する。
2. preflight processは自分の`OMPI_COMM_WORLD_RANK`とhostnameだけを出力し、
   job masterが完全なrank-host mappingを取得する。
3. job masterが`jobs/pegasus/prepare_wave.py`を実行する。
4. `prepare_wave.py`がrun ID、wave、expected node count、rank-host mapping、
   hostname重複、既存output、config/binary/source/runtime provenanceを検査する。
5. 新規campaignではjob masterだけが`effective-config.json`、
   `run-metadata.json`、`runtime-environment.json`を排他的に作成する。追加wave
   では既存hashとimmutable provenanceを検査する。
6. job masterだけが`wave-metadata.json`を排他的に作成する。
7. preflight成功後だけ、本測定用`mpirun`で`run_node.sh`を起動する。

single-node実行でもjob masterが同じpreflight処理を行う。preflight failure時は
本測定用`mpirun`を開始しない。同一run IDへwaveを追加できるのは、effective
config、executables manifest、runtime environment、Git commit、dirty state、
およびdirty時のsource hashがすべて一致する場合だけである。

production runはclean worktreeを必須とし、dirtyならjob masterが本測定前に
拒否する。smoke/pilotでdirtyを許す場合は、完全なGit diff hashまたはsource
snapshot hashをrun provenanceへ保存する。

## Node-local scratchと回収

各node processは測定中のraw result、stdout、stderr、telemetryを自nodeの`/scr`へ
書く。一時pathには少なくともjob ID、run ID、wave、hostnameを含め、同時実行と
再実行で衝突しないようにする。

各node processが書いてよいshared outputは次だけである。

```text
results/<run-id>/waves/<wave>/nodes/<hostname>/
```

`run_node.sh`はtrapを使い、正常終了と異常終了の双方で次を行う。

1. background telemetryを停止する。
2. node statusを確定する。
3. raw result、log、telemetryを自分のnode directoryへ回収する。
4. collection failureをstatusへ記録する。

raw resultを削除または上書きしない。既存wave/node directoryとの衝突は失敗と
する。初期版ではBeeOND、`USE_PMEM_BEEOND`、`USE_DEVDAX`、`USE_MEM`を
使用しない。

node内では`run_suite.py`だけがnode-level raw fileを排他的に作成・appendする。
benchmark executableは`--output -`でmachine-readable stdoutを返し、raw fileへ
直接書かない。runnerはstdout validationとsynthetic failure row生成を担当する。

## Node failure isolationとjob最終status

`run_node.sh`はbenchmark failure、verification failure、tool failureを
`node-status.json`へ記録し、trapでraw result、stdout、stderr、telemetryを回収
する。他nodeの回収を本測定用`mpirun`から強制終了させないため、必要なstatusと
成果物の回収に成功したnode runnerは、benchmark/verification/tool failureが
あっても原則exit code 0で終了する。

preflight failure、node directory作成不能、`node-status.json`作成不能、result
回収不能など、成果物を信頼して回収できないfatal infrastructure failureは非ゼロ
終了してよい。

本測定用`mpirun`終了後、PBS job masterは必ず`collect_results.py`を実行する。
collectorはpreflightで確定した全expected nodeの`node-status.json`を検査する。
missing node、benchmark failure、verification failure、collection failureが1件でも
あれば、raw dataとnode logを保持したままjob全体を非ゼロ終了させる。全nodeの
statusと回収結果が成功した場合だけjob全体を成功とする。collectorはartifactの
一覧とhash manifestを作成するが、raw dataを集計、変更、削除、filterしない。

## CPU thread環境

productionでは次を設定JSONから明示的に設定し、node metadataとtelemetryへ
記録する。

```text
OMP_NUM_THREADS
OMP_PROC_BIND
OMP_PLACES
OMP_DYNAMIC=FALSE
MKL_NUM_THREADS
MKL_DYNAMIC=FALSE
MKL_THREADING_LAYER
OPENBLAS_NUM_THREADS
```

標準要求は`OMP_NUM_THREADS=48`だが、これは全backendが48threadを使うという
意味ではない。cuRANDの`cpu-std-random-serial`とThrustの
`cpu-stl-serial`はeffective thread数1のserial production baselineであり、
48-core CPU性能と表現しない。requested/effective thread数とparallelismの
schemaは`RESULT_SCHEMA.md`に従う。

## GPU telemetry

実行前後に次を保存する。

```bash
nvidia-smi -q
```

実行中の基本commandは次とする。

```bash
nvidia-smi dmon -s pucv -d 1 -o T --format csv,nounit
```

- shell redirectionでraw dmon outputを必ず保存する。
- background PIDを記録し、trapで停止する。
- parserは固定column番号でなくheader名を使う。
- 未知columnをraw outputから削除しない。
- power、temperature、utilization、processor clock、memory clock、power
  violation、thermal violationを利用可能なheaderから記録する。
- GPU UUIDを保存する。
- `--format`が利用できない場合はplain-text outputへfallbackし、その事実を
  `telemetry_status`へ記録する。
- 欠落column、command failure、parse failureはnon-fatalとし、statusへ記録する。
- telemetry failureを理由にbenchmark raw resultを失敗・削除しない。

## CPU telemetry

各nodeで次を保存する。

- hostname
- date
- `lscpu`
- `/proc/cpuinfo`
- `numactl --hardware`（利用可能な場合）
- `taskset -cp $$`
- `module list`
- 使用compilerのversion
- `OMP_NUM_THREADS`
- `OMP_PROC_BIND`
- `OMP_PLACES`
- `OMP_DYNAMIC`
- `MKL_NUM_THREADS`
- `MKL_DYNAMIC`
- `MKL_THREADING_LAYER`
- `OPENBLAS_NUM_THREADS`

利用可能な場合だけ`turbostat`、または読み取り可能なCPU
frequency sysfs情報を保存する。取得不能はnon-fatalとし、`telemetry_status`へ
記録する。`sudo`、`cpupower`、CPU governor変更を行わない。

## Telemetry timestamp metadata

各telemetry streamには`telemetry_start_timestamp_utc`、
`telemetry_end_timestamp_utc`、`sample_interval_sec`、`timezone`、`utc_offset`を
保存し、さらに`midnight_rollover_count`を保存する。timestampはUTC・millisecond
精度・末尾`Z`とし、timezoneとUTC offsetは取得時のlocal-time解釈を監査するために
併記する。

`nvidia-smi dmon`の`-o T`出力は時刻だけを持つため、raw outputに加え、
開始日、timezone、UTC offset、およびlocal midnight rolloverの検出結果を保存する。
parserはこれらから各sampleのUTC時刻を復元し、raw resultの
`measurement_start_timestamp`から`measurement_end_timestamp`までの区間と
重ねて相関する。単一timestampへの最近傍対応だけを測定区間の代表値にしない。
復元不能や時計の逆行はnon-fatalなparse failureとして`telemetry_status`へ記録し、
raw telemetryとbenchmark resultを保持する。

## Smoke testとproduction

### Smoke test

- `gpu` queueへのbatch job、1〜2ノード、1時間以内
- 小さいproblem size
- build確認、example実行、benchmark 1 trial
- `compute-sanitizer`
- result/log/telemetry回収
- dirty sourceは必要なsource provenance hashを保存した場合だけ許可

### Production

- 人間が利用可能queueとbudgetを確認する
- clean worktreeを必須とする
- 最低5ノード、可能なら8ノード
- 別時間帯のwaveを追加可能にする
- account、queue、module versionを設定JSONへ入力する
- job投入は人間だけが行う

## Buildとrunの分離

実装済みの`jobs/pegasus/`成果物は次とする。

```text
jobs/pegasus/README.md
jobs/pegasus/pegasus.json.example
jobs/pegasus/render_job.py
jobs/pegasus/prepare_wave.py
jobs/pegasus/build_cpu_cuda.sh
jobs/pegasus/build_openacc.sh
jobs/pegasus/run_benchmarks.pbs.in
jobs/pegasus/run_node.sh
jobs/pegasus/collect_results.py
```

`render_job.py`はaccount、queue、MPI version等の必須値がなければ失敗する。
buildとbenchmark jobを分離し、batch benchmark job内ではcompileしない。

- `prepare_wave.py`: preflight rank-host mappingと全provenanceを検証し、job
  masterだけが所有するcampaign/wave metadataを排他的に作成する。
- `build_cpu_cuda.sh`: GCC/G++/nvcc系build treeを作る。
- `build_openacc.sh`: NVHPC/OpenACC系build treeを作る。
- build script自身はscheduler commandを実行しない。
- 人間が適切な計算node上でbuild scriptを実行する。
- benchmark jobは事前build済みでmanifestにhash記録されたbinaryだけを使う。
