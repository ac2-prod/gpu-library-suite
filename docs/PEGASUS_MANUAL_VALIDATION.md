# Pegasus手動検証ガイド

## 目的と境界

この文書は、人間がPegasus上で実施するbuild、smoke、production検証のchecklist
である。規範的なresource、PBS、module、scratch、telemetry、failure処理は
[`PEGASUS_EXECUTION.md`](PEGASUS_EXECUTION.md)を正本とする。このrepositoryの
renderer、build script、runnerはjobの投入・照会・削除やaccount確認を実行しない。

local検証ではCUDA GPU、NVHPC、Pegasus jobを実行していない。そのため以下の全項目
は、実機で確認して記録されるまで未検証である。

## 1. 人間によるsite確認

- 利用可能なaccount、queue、budget、node上限、walltimeを確認する。
- CPU/CUDA build、OpenACC build、benchmark runtimeに必要なmodule名とversionを
  個別に確認する。
- OpenMPI version、CUDA Toolkit root/version、NVHPCが選択するCUDA Toolkit、
  GPU target、shared result rootを確認する。
- `pegasus.json.example`をコピーし、全placeholderを実値へ置換する。credentialや
  private keyは保存しない。
- 8個のCPU runtime変数を明示し、threads/process × processes/nodeが物理core数を
  超えないことを確認する。

確認commandとjob投入commandは人間だけが実行する。command output、実行日時、
実行者、選択理由をrepository外の運用記録へ保存する。

## 2. buildとmanifest

1. `build_cpu_cuda.sh`でCPU/CUDA profileをbuildする。
2. `build_openacc.sh`でNVHPC/OpenACC profileを別treeへbuildする。
3. 両treeのconfigure summaryを保存し、optional targetのdisable理由を確認する。
4. OpenACC build metadataでCMakeとNVHPCのCUDA Toolkit path/version、
   `NVHPC_CUDA_HOME`、compiler flagsが一致することを確認する。
5. 各treeの`gpu_suite_partial_manifest`を生成し、`merge_manifests.py`でmergeする。
6. 全binary hash、Git commit、clean state、Release build、target/profile/backend
   variantを照合する。
7. enabledになった全teaching exampleを実行し、各libraryのAPI errorと数値結果を
   確認する。

異なるprovider/profileの衝突、Toolkit不一致、dirty production buildを回避して
続行してはならない。

## 3. render前確認

- 最初は`configs/pilot.json`を使用し、値をproduction確定値と扱わない。
- `render_job.py`の入力config、manifest、両build metadata、run ID、wave、system
  labelが今回のcampaign用であることを確認する。
- `pbs_stdout`と`pbs_stderr`の親directoryを人間が事前に作成済みであることを
  確認する。rendererは存在を検査するだけで暗黙作成しない。
- 生成PBSを全文確認し、`bash -n`を実行する。
- PBSがbenchmark job内でcompileせず、`benchmark_runtime_modules`だけをloadし、
  人間が選択したaccount/queue/pathを使用することを確認する。
- job投入は内容確認後に人間が行う。

## 4. smoke検証

最初のsmokeは小規模かつ短時間とし、次を全て確認する。

- preflight rank-host mappingがexpected node数と一致し、hostname重複がない。
- job masterだけがcampaign/wave metadataを排他的に作成する。
- `runtime-environment.json`にcanonical module/load順、path、driver、
  CUDA/NVHPC/CPU library、全binaryのaddress-free canonical dependency、解決
  library path、8 CPU変数があり、hashがmetadata間で一致する。
- waveごとの`job-master/runtime-environment-evidence.json`に元のmodule list、
  command diagnostic、GPU queryおよびaddress/line順を含むraw `ldd`が残り、その
  hashとcanonical runtime/manifestへの参照が`wave-metadata.json`と一致する。
- prebuilt binaryのruntime検査ではdriver、CUDA runtime、shared libraryを必須とし、
  `nvcc`、`nvc`、`nvc++` commandはoptional metadataとして欠落理由を記録する。
  site規則でcompiler commandを必須化する場合だけ
  `require_runtime_compilers=true`にし、その理由を運用記録へ残す。
- 各node自身が取得したGPU name、UUID、NVIDIA package driverと、node-local
  `libcudart`から取得したCUDA Driver API/Runtime versionが
  `node-metadata.json`へ保存され、成功CUDA/OpenACC raw rowの4項目と一致し、別nodeの
  identityを流用していない。
- 異なるallocationの2 waveではhostname、scheduler job ID、GPU UUIDが各wave/node
  provenanceに残る一方、software identityが同じならcanonical runtime hashが一致し、
  raw evidence sidecarの差だけで追加waveを拒否しない。
- CPU、CUDA、OpenACCが各node block内で指定順序によりinterleaveされる。
- raw rowのverification、UTC millisecond timestamp、requested/effective threads、
  compiler/library/GPU identityを確認する。
- cuFFT primary denominatorがthreaded FFTWであり、欠落時にserialへfallback
  しない。
- cuRAND/Thrust serial backendはeffective 1 threadであり、cuRANDは同じ
  distribution/output typeだが異なるalgorithmの比較として記録される。
- raw dmon output、timezone/offset、midnight rollover count、trial intervalとの
  相関を確認する。
- collectorが全expected node statusを検査し、artifact manifestのhashが実fileと
  一致する。

## 5. 異常系smoke

安全なsmoke campaignで意図的に1 nodeのbenchmark/verificationを失敗させる。

- failure nodeが`node-status.json`と取得可能なartifactを回収する。
- 回収成功nodeは原則0で終了し、他nodeの測定・回収が継続する。
- job master collectorは全node回収後にjob全体を非ゼロとする。
- missing status、shared directory作成不能、copy不能はfatal infrastructure
  failureとして区別される。
- completed raw trialが保持され、先頭の未解決trialだけattempted failure、残りは
  unattempted skippedとなり、trial index重複がない。

## 6. production移行判定

- smokeと異常系smokeの全required項目が成功している。
- worktreeがcleanで、全waveのconfig、manifest/binary、Git、build、runtime
  environment provenanceが一致する。
- pilot結果を基に同じ`configs/benchmark.json`を更新し、schema versionやfilename
  を変更していない。
- 5 node以上、可能なら8 nodeでordering balanceを得る計画になっている。
- production後に`validate_results.py`を通し、validation失敗を残したままaggregate
  やperformance報告へ進まない。

## 実行記録に残す項目

各実行について、run ID、wave、scheduler job ID、UTC開始/終了時刻、human-verified
account/queue/module、Git commit/dirty state、config/manifest/runtime hash、node数、
成功・failure・skipped件数、verification failure、collection failure、telemetry
status、sanitizer結果、validation/aggregation結果を記録する。

現時点の実行状況は[`VALIDATION_REPORT.md`](VALIDATION_REPORT.md)に記載する。
