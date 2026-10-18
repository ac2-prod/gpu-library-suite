# rocThrust C/C++

CPU・HIP・OpenMP targetでFP64の`sum(values[i]^2)`を実装しています。
Sirius実機build、教材3本、数値smoke全96件の成功を確認済みで、reduceの誤差は0でした。
[検証したsource snapshotと保存ログ](../../../jobs/sirius/README.md#focused-numerical-smoke)を参照してください。
確認したCPU backendは逐次STLです。性能と後続のソース変更は未検証です。
教材の既定問題は`1 << 24`個、全要素1.0で、期待値は要素数です。

CPU既定はSTL、`GPU_SUITE_CPU_THRUST_BACKEND=OPENMP`を明示したbuildでは
別backendの`cpu-openmp`です。HIPは同期的な`thrust::hip_rocprim::par`を使います。

OpenMP教材だけは2翻訳単位です。`openmp_rocthrust.cpp`をOpenMP targetで、
正準名`rocthrust_wrapper.hip`をHIPで別々にコンパイルし、両objectを明示的に
リンクします。wrapperはC ABIを持ち、例外を外へ出さず、OpenMP所有bufferを
解放しません。呼出し側が実offloadとmappingのHIP device対応を確認します。

benchmarkのcompute scopeでは入力allocationを保持し、同期reductionを計測します。
end-to-endはallocation・転送・reductionと結果取得までを含み、明示的な解放は
計測外です。入力は不変で、各reductionの初期値は0.0です。要素数との数値比較、
非有限値、byte数とiterator差分型のoverflowを検査します。

2翻訳単位は別々にコンパイルして明示的にリンクします。
`HIP_TARGET_FLAGS`と`OMP_TARGET_FLAGS`には、外部で選択した同一GPU targetを
明示してください。GPUの自動探索に任せません。

```sh
hipcc -std=c++17 $HIP_TARGET_FLAGS -c examples/rocthrust_wrapper.hip -o rocthrust_wrapper.o
"$OMP_CXX" -std=c++17 $OMP_TARGET_FLAGS $HIP_INCLUDE_FLAGS -c examples/openmp_rocthrust.cpp -o openmp_rocthrust.o
"$OMP_CXX" $OMP_TARGET_FLAGS openmp_rocthrust.o rocthrust_wrapper.o $HIP_LINK_FLAGS -o openmp_rocthrust
```

CMakeでも最終リンクにはOpenMP側のC++コンパイラを使用します。
configure時には正準wrapperとOpenMP callerの2翻訳単位を実際にコンパイル・
リンクし、生成した実行ファイルは起動しません。検査ログはbuild tree内の
`CMakeFiles/GPU_SUITE_ROCTHRUST_OPENMP_LINKS/output.log`へ保存します。

ファイル一覧は[英語版](README.md)を参照してください。
規範は[仕様書](../../../docs/PROJECT_SPECIFICATION.md)と
[benchmark protocol](../../../docs/BENCHMARK_PROTOCOL.md)です。
ローカルのfake実行は実GPUの成功として扱いません。
