"""AMD result/runner contracts, using records only (no ROCm or remote execution)."""
import copy
import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

from gpu_suite.aggregation import aggregate_results
from gpu_suite.config import load_config, validate_config
from gpu_suite.hashing import sha256_file
from gpu_suite.manifest import artifact_id, merge_entries, validate_build_metadata
from gpu_suite.ordering import implementation_order
from gpu_suite.plotting import build_plot_metadata
from gpu_suite.results_io import ExclusiveRawWriter, load_raw_results
from gpu_suite.runner import (build_schedule, core_raw_parameters, dry_run_document,
                              synthetic_result, validate_subprocess_record)
from gpu_suite.schema import validate_raw_result
from gpu_suite.strict_json import dumps
from gpu_suite.validation import validate_campaign, validate_rocfft_provenance
from test_manifest import manifest_entry
from test_schema import success_record

ROOT = Path(__file__).resolve().parents[2]
ZERO = '0' * 64


def amd_config():
    return load_config(ROOT / 'jobs/sirius/rocfft-smoke.json')


def amd_entry(implementation):
    target = {'cpu': 'fft_cpu_bench', 'hip': 'fft_gpu_bench', 'openmp': 'openmp_rocfft_bench'}[implementation]
    entry = manifest_entry(target=target, role='benchmark', path='/fixture/' + target)
    entry.update(library='rocfft', implementation=implementation, build_profile=implementation,
                 compiler_language={'cpu': 'c', 'hip': 'hip', 'openmp': 'cxx'}[implementation],
                 supported_cpu_backends=['cpu-fftw-serial', 'cpu-fftw-threaded'] if implementation == 'cpu' else [])
    entry['artifact_id'] = artifact_id(entry)
    return entry


def campaign():
    config = amd_config()
    entries = [amd_entry(impl) for impl in ('cpu', 'hip', 'openmp')]
    manifest = {'manifest_schema_version': 1, 'entries': merge_entries([entries])}
    context = dict(config=config, cpu_threads=1, device=0, hostname='node0', node_index=0,
                   run_id='run-1', wave=0, scheduler=None, scheduler_job_id=None,
                   system_label='local', implementation_order=implementation_order(0, 0, 'amd'))
    schedule = build_schedule(config, manifest, context)
    records = []
    for item in schedule:
        for trial in range(item['scope_settings']['trials']):
            row = success_record(trial)
            impl = item['series']['implementation']
            row.update(benchmark='rocfft', implementation=impl, scope=item['scope'],
                       implementation_order=['cpu', 'hip', 'openmp'],
                       parameters=dict(item['parameters']), problem_size=item['parameters']['nfft'],
                       secondary_size=item['parameters']['batch'], warmup=item['scope_settings']['warmup'],
                       repeat=item['scope_settings']['repeat'], binary_sha256='1' * 64)
            row['parameters']['runtime'] = {'build_metadata_sha256': '2' * 64}
            if impl == 'cpu':
                row.update(cpu_backend='cpu-fftw-serial', library_name='cpu-fftw-serial',
                           cpu_threads_requested=1, cpu_threads_effective=1, cpu_parallelism='serial')
            else:
                row.update(cpu_backend=None, cpu_backend_role=None, cpu_threads_requested=None,
                           cpu_threads_effective=None, cpu_parallelism=None, device_id=0,
                           gpu_name='fixture only', gpu_uuid=None, library_name='rocfft', library_version='fixture')
                row['parameters']['runtime'].update(hip_runtime_version=70200000, hip_driver_version=70200000)
            records.append(validate_raw_result(row))
    return config, manifest, context, schedule, records


class RocfftToolsTests(unittest.TestCase):
    def test_vendor_config_and_family_rejections(self):
        config = amd_config()
        self.assertEqual(set(config['benchmarks']), {'rocfft'})
        for mutation in ('vendor', 'source_language', 'implementation', 'transform', 'precision'):
            changed = copy.deepcopy(config)
            if mutation == 'vendor': changed['vendor'] = 'nvidia'
            elif mutation == 'source_language': changed['source_language'] = 'fortran'
            elif mutation == 'implementation': changed['benchmarks']['rocfft']['series'][1]['implementation'] = 'cuda'
            elif mutation == 'transform': changed['benchmarks']['rocfft']['cases'][0]['parameters']['transform'] = 'inverse'
            else: changed['benchmarks']['rocfft']['precision'] = 'fp64'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_config(changed)

    def test_old_nvidia_config_and_record_are_unchanged(self):
        from gpu_suite.config import BENCHMARKS
        self.assertEqual(BENCHMARKS, ("cufft", "cublas", "cusparse", "cusolver", "curand", "thrust"))
        for name in ('benchmark.json', 'pilot.json'):
            path = ROOT / 'configs' / name
            config = load_config(path)
            self.assertNotIn('vendor', config)
            self.assertEqual(len(config['benchmarks']), 6)
        old = success_record()
        self.assertEqual(validate_raw_result(old), old)
        self.assertEqual(implementation_order(0, 0), ('cpu', 'cuda', 'openacc'))

    def test_amd_order_has_all_six_permutations(self):
        orders = {implementation_order(index, 0, 'amd') for index in range(6)}
        self.assertEqual(len(orders), 6)
        self.assertTrue(all(set(order) == {'cpu', 'hip', 'openmp'} for order in orders))

    def test_all_three_manifests_and_invalid_profile(self):
        entries = [amd_entry(impl) for impl in ('cpu', 'hip', 'openmp')]
        self.assertEqual(len(merge_entries([entries])), 3)
        for entry in entries:
            bad = dict(entry, build_profile='cpu-cuda')
            bad['artifact_id'] = artifact_id(bad)
            with self.assertRaises(ValueError): merge_entries([[bad]])

    def test_jsonl_csv_roundtrip_with_hip_metadata_and_null_cuda(self):
        records = campaign()[4]
        with tempfile.TemporaryDirectory() as directory:
            for fmt in ('jsonl', 'csv'):
                path = Path(directory) / ('raw.' + fmt)
                with ExclusiveRawWriter(path, fmt) as writer:
                    for row in records: writer.write(row)
                self.assertEqual(load_raw_results(path), records)
        for row in records:
            self.assertIsNone(row['cuda_runtime_version'])
            self.assertIsNone(row['cuda_driver_version'])

    def test_gpu_metadata_cannot_be_cuda_or_missing_versions(self):
        row = next(row for row in campaign()[4] if row['implementation'] == 'hip')
        for field in ('cuda_runtime_version', 'cuda_driver_version', 'runtime'):
            bad = copy.deepcopy(row)
            if field == 'runtime': bad['parameters']['runtime'].pop('hip_runtime_version')
            else: bad[field] = 'not HIP'
            with self.subTest(field=field), self.assertRaises(ValueError): validate_raw_result(bad)

    def test_runtime_is_not_mathematical_key(self):
        row = campaign()[4][0]
        self.assertEqual(core_raw_parameters('rocfft', row['parameters']),
                         {key: row['parameters'][key] for key in ('nfft', 'batch', 'transform')})

    def test_runner_dry_run_commands_all_implementations_and_scopes(self):
        config, manifest, context, schedule, _ = campaign()
        doc = dry_run_document(config, manifest, schedule, context, ZERO, ZERO)
        self.assertEqual(len(doc['commands']), 12)
        self.assertEqual({item['series']['implementation'] for item in schedule}, {'cpu', 'hip', 'openmp'})
        for item in schedule:
            self.assertIn('--batch', item['argv'])
            self.assertIn('cpu,hip,openmp', item['argv'])
            self.assertIsNone(item['prerequisite_failure'])

    def test_missing_artifact_stays_prerequisite_skip(self):
        config, _, context, _, _ = campaign()
        schedule = build_schedule(config, {'entries': []}, context)
        records = [synthetic_result(item, context, ZERO, ZERO, {}, trial, False,
                                   'prerequisite', 'not built', None, None, None)
                   for item in schedule for trial in range(item['scope_settings']['trials'])]
        summary = validate_campaign(records, config, ZERO, {'entries': []})
        self.assertEqual(summary['validation_status'], 'failure')
        self.assertEqual(summary['status_counts'], {'skipped': len(records)})
        for row in records:
            self.assertFalse(row['git_metadata_available'])
            for field in ('git_commit', 'git_dirty', 'git_diff_sha256', 'source_snapshot_sha256'):
                self.assertIsNone(row[field])

    def test_missing_cpu_preserves_source_but_not_artifact_identity(self):
        config, manifest, context, _, _ = campaign()
        manifest['entries'] = [e for e in manifest['entries'] if e['implementation'] != 'cpu']
        schedule = build_schedule(config, manifest, context)
        for diff, snapshot in ((ZERO, None), (None, '3' * 64), (ZERO, '3' * 64)):
            for item in schedule:
                if item['entry'] is not None:
                    continue
                row = synthetic_result(item, context, ZERO, ZERO, {}, 0, False,
                                       'prerequisite', 'not built', None, diff, snapshot)
                self.assertTrue(row['git_metadata_available'])
                self.assertEqual((row['git_commit'], row['git_dirty']), ('abc', True))
                self.assertEqual(row['git_diff_sha256'], diff)
                self.assertEqual(row['source_snapshot_sha256'], snapshot)
                self.assertEqual(row['binary_sha256'], ZERO)
                self.assertEqual((row['compiler'], row['compiler_version']), ('unavailable', 'unavailable'))
                self.assertNotIn('runtime', row['parameters'])
                self.assertEqual((row['status'], row['attempted'], row['exit_code']), ('skipped', False, None))

    def test_missing_cpu_rejects_mixed_source_before_execution(self):
        config, manifest, context, _, _ = campaign()
        manifest['entries'] = [e for e in manifest['entries'] if e['implementation'] != 'cpu']
        for mutation in ('commit', 'dirty', 'unavailable'):
            changed = copy.deepcopy(manifest)
            entry = changed['entries'][0]
            if mutation == 'commit': entry['git_commit'] = 'different-source'
            elif mutation == 'dirty': entry['git_dirty'] = False
            else: entry.update(git_metadata_available=False, git_commit=None, git_dirty=None)
            with self.subTest(mutation=mutation), mock.patch('gpu_suite.runner.subprocess.run') as launch:
                with self.assertRaisesRegex(ValueError, 'mixed AMD source provenance'):
                    build_schedule(config, changed, context)
                launch.assert_not_called()

    def test_missing_cpu_keeps_unavailable_source_unknown(self):
        config, manifest, context, _, _ = campaign()
        manifest['entries'] = [e for e in manifest['entries'] if e['implementation'] != 'cpu']
        for entry in manifest['entries']:
            entry.update(git_metadata_available=False, git_commit=None, git_dirty=None)
        for item in build_schedule(config, manifest, context):
            if item['entry'] is None:
                row = synthetic_result(item, context, ZERO, ZERO, {}, 0, False,
                                       'prerequisite', 'not built', None, ZERO, '3' * 64)
                self.assertFalse(row['git_metadata_available'])
                for field in ('git_commit', 'git_dirty', 'git_diff_sha256', 'source_snapshot_sha256'):
                    self.assertIsNone(row[field])

    def test_missing_cpu_stays_prerequisite_after_prior_failure_in_every_order(self):
        from gpu_suite.runner import execute_schedule
        config, manifest, context, _, _ = campaign()
        manifest['entries'] = [e for e in manifest['entries'] if e['implementation'] != 'cpu']
        metadata = {'2' * 64: {language: {name: manifest['entries'][0][name] for name in
                              ('compiler', 'compiler_version', 'global_configure_flags')}
                              for language in ('hip', 'cxx')}}
        for node_index in range(6):
            with self.subTest(node_index=node_index), tempfile.TemporaryDirectory() as directory:
                context.update(node_index=node_index,
                               implementation_order=implementation_order(node_index, 0, 'amd'))
                schedule = build_schedule(config, manifest, context)
                output = Path(directory) / 'raw.jsonl'
                with mock.patch('gpu_suite.runner._validate_amd_dependencies'), \
                     mock.patch('gpu_suite.runner.sha256_file', return_value='1' * 64), \
                     mock.patch('gpu_suite.runner.subprocess.run',
                                return_value=SimpleNamespace(returncode=1, stdout='', stderr='')) as launch:
                    self.assertEqual(execute_schedule(schedule, context, output, ZERO, ZERO,
                                                      metadata, ZERO, None), 1)
                self.assertEqual(launch.call_count, 0 if context['implementation_order'][0] == 'cpu' else 1)
                self.assertTrue(all('fft_cpu_bench' not in call.args[0][0] for call in launch.call_args_list))
                rows = load_raw_results(output)
                summary = validate_campaign(rows, config, ZERO, manifest)
                self.assertEqual(summary['validation_status'], 'failure')
                self.assertTrue(all(row['failure_origin'] == 'prerequisite' and not row['attempted']
                                    for row in rows if row['implementation'] == 'cpu'))
                self.assertTrue(all(row['failure_origin'] == 'prior-failure'
                                    for row in rows if row['implementation'] != 'cpu' and not row['attempted']))

    def test_missing_cpu_cli_records_validate_in_jsonl_and_csv(self):
        import run_suite
        from gpu_suite.runner import _expected_verification

        for fmt in ('jsonl', 'csv'):
            for gpu_status in ('success', 'failure'):
                with self.subTest(format=fmt, gpu_status=gpu_status), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    config, _, context, _, _ = campaign()
                    config.update(output_format=fmt, continue_on_failure=True)
                    config['benchmarks']['rocfft']['cases'] = config['benchmarks']['rocfft']['cases'][:1]
                    for scope in config['benchmarks']['rocfft']['cases'][0]['scopes'].values():
                        scope['trials'] = 1
                    config_path = root / 'config.json'
                    config_path.write_text(dumps(config), encoding='utf-8')
                    config_hash = sha256_file(config_path)
                    dependency = root / 'fixture-library'
                    dependency.write_bytes(b'fixture only, never loaded')
                    entries, metadata, metadata_paths = [], {}, []
                    for impl in ('hip', 'openmp'):
                        entry = amd_entry(impl)
                        binary = root / entry['target_name']
                        binary.write_bytes(b'fixture only, never executed')
                        entry.update(executable_path=str(binary), binary_sha256=sha256_file(binary))
                        meta = {name: entry[name] for name in ('build_type', 'build_profile',
                                'git_metadata_available', 'git_commit', 'git_dirty')}
                        meta.update(vendor='amd', dependencies={name: {'path': str(dependency),
                                    'sha256': sha256_file(dependency)} for name in ('hip', 'rocfft')})
                        meta[entry['compiler_language']] = {name: entry[name] for name in
                                    ('compiler', 'compiler_version', 'global_configure_flags')}
                        meta_path = root / (impl + '-metadata.json')
                        meta_path.write_text(dumps(meta), encoding='utf-8')
                        entry['build_metadata_sha256'] = sha256_file(meta_path)
                        entry['artifact_id'] = artifact_id(entry)
                        entries.append(entry)
                        metadata[entry['build_metadata_sha256']] = meta
                        metadata_paths.append(meta_path)
                    manifest = {'manifest_schema_version': 1, 'entries': merge_entries([entries])}
                    manifest_path = root / 'manifest.json'
                    manifest_path.write_text(dumps(manifest), encoding='utf-8')
                    schedule = build_schedule(config, manifest, context)
                    expected_calls = [item['argv'] for item in schedule if item['entry'] is not None]
                    responses = []
                    for index, item in enumerate(item for item in schedule if item['entry'] is not None):
                        row = synthetic_result(item, context, config_hash, ZERO, metadata, 0, True,
                                               'subprocess', 'fixture failure', 1, ZERO, '3' * 64)
                        if gpu_status == 'success':
                            row.update({name: success_record()[name] for name in
                                        ('measurement_start_timestamp', 'measurement_end_timestamp',
                                         'elapsed_total_sec', 'elapsed_sec', 'exit_code', 'failure_origin',
                                         'status', 'message', 'verification_status')})
                            thresholds, metrics, primary = _expected_verification(item, context)
                            row.update(verification_thresholds=thresholds,
                                       verification_metrics={name: 0.0 for name in metrics},
                                       verification_primary_metric=primary, library_version='fixture',
                                       gpu_name='fixture only')
                            row['parameters']['runtime'].update(hip_runtime_version=70200000,
                                                                 hip_driver_version=70200000)
                        response_path = root / ('fixture-stdout-{0}.{1}'.format(index, fmt))
                        with ExclusiveRawWriter(response_path, fmt) as writer:
                            writer.write(row)
                        responses.append(SimpleNamespace(returncode=0 if gpu_status == 'success' else 1,
                                                         stdout=response_path.read_text(), stderr=''))
                    output = root / ('raw.' + fmt)
                    argv = ['--config', str(config_path), '--manifest', str(manifest_path),
                            '--output', str(output), '--run-id', 'run-1', '--system-label', 'local',
                            '--wave', '0', '--node-index', '0', '--hostname', 'node0',
                            '--runtime-environment-sha256', ZERO, '--git-diff-sha256', ZERO,
                            '--source-snapshot-sha256', '3' * 64]
                    for path in metadata_paths:
                        argv.extend(['--build-metadata', str(path)])
                    with mock.patch('gpu_suite.runner.subprocess.run', side_effect=responses) as launch:
                        with contextlib.redirect_stderr(io.StringIO()) as stderr:
                            self.assertEqual(run_suite.main(argv), 1)  # CPU remains unavailable.
                        self.assertEqual(stderr.getvalue(), '')
                        self.assertEqual([call.args[0] for call in launch.call_args_list], expected_calls)
                    rows = load_raw_results(output)
                    summary = validate_campaign(rows, config, config_hash, manifest)
                    self.assertEqual(summary['validation_status'], 'failure')
                    self.assertEqual(summary['status_counts'], {'skipped': 2, gpu_status: 4})
                    cpu = [row for row in rows if row['implementation'] == 'cpu']
                    self.assertTrue(all(not row['attempted'] and row['failure_origin'] == 'prerequisite' for row in cpu))
                    self.assertTrue(all(row['git_diff_sha256'] == ZERO and
                                        row['source_snapshot_sha256'] == '3' * 64 for row in rows))
                    for field, value in (('git_commit', 'other'), ('git_diff_sha256', '4' * 64),
                                         ('source_snapshot_sha256', '4' * 64)):
                        changed = copy.deepcopy(rows)
                        changed[0][field] = value
                        with self.assertRaisesRegex(ValueError, 'mixed AMD provenance'):
                            validate_campaign(changed, config, config_hash, manifest)

    def test_complete_campaign_and_aggregation_plot(self):
        config, manifest, _, _, records = campaign()
        self.assertEqual(validate_campaign(records, config, ZERO, manifest)['validation_status'], 'pass')
        aggregate, _ = aggregate_results(records, config)
        self.assertEqual({r['comparison'] for r in aggregate if r['comparison']},
                         {'cpu/hip', 'cpu/openmp'})
        metadata = build_plot_metadata(aggregate, ZERO, raw_records=records)
        self.assertEqual({item['benchmark'] for item in metadata['series']}, {'rocfft'})
        self.assertTrue(any('OpenMP target' in item['label'] for item in metadata['series']))

    def test_mismatched_config_source_binary_compiler_and_build_rejected(self):
        config, manifest, _, _, records = campaign()
        for field in ('config_sha256', 'runtime_environment_sha256', 'source_snapshot_sha256',
                      'git_diff_sha256', 'git_commit', 'binary_sha256', 'compiler',
                      'compiler_version', 'global_configure_flags', 'build_metadata_sha256'):
            changed = copy.deepcopy(records)
            if field == 'build_metadata_sha256': changed[0]['parameters']['runtime'][field] = '9' * 64
            else: changed[0][field] = '9' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_campaign(changed, config, ZERO, manifest)

    def test_runtime_library_versions_rejected_despite_equal_math(self):
        config, manifest, _, _, records = campaign()
        index = next(i for i, row in enumerate(records) if row['implementation'] == 'hip')
        for field in ('hip_runtime_version', 'hip_driver_version', 'library_version'):
            changed = copy.deepcopy(records)
            if field == 'library_version': changed[index][field] = 'different-library'
            else: changed[index]['parameters']['runtime'][field] += 1
            self.assertEqual(core_raw_parameters('rocfft', records[index]['parameters']),
                             core_raw_parameters('rocfft', changed[index]['parameters']))
            with self.subTest(field=field):
                with self.assertRaises(ValueError): validate_campaign(changed, config, ZERO, manifest)
                with self.assertRaises(ValueError): aggregate_results(changed, config)

    def test_resolved_library_and_build_conditions_are_hash_bound(self):
        from gpu_suite.hashing import deterministic_json_sha256
        metadata = {'dependencies': {'rocfft': {'path': '/fixture/librocfft.so', 'sha256': '1' * 64}},
                    'hip': {'global_configure_flags': '-O2', 'architectures': ['fixture']}}
        original = deterministic_json_sha256(metadata)
        config, manifest, _, _, records = campaign()
        for entry in manifest['entries']: entry['build_metadata_sha256'] = original
        for row in records: row['parameters']['runtime']['build_metadata_sha256'] = original
        self.assertEqual(validate_campaign(records, config, ZERO, manifest)['validation_status'], 'pass')
        for kind in ('path', 'sha256', 'flags'):
            altered = copy.deepcopy(metadata)
            if kind == 'flags': altered['hip']['global_configure_flags'] = '-O3'
            else: altered['dependencies']['rocfft'][kind] = 'different'
            changed = copy.deepcopy(records)
            changed[0]['parameters']['runtime']['build_metadata_sha256'] = deterministic_json_sha256(altered)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_campaign(changed, config, ZERO, manifest)

    def test_runner_checks_runtime_build_binding_and_exact_parameters(self):
        from gpu_suite.runner import _expected_verification
        config, _, context, schedule, records = campaign()
        item, row = schedule[0], copy.deepcopy(records[0])
        thresholds, metrics, primary = _expected_verification(item, context)
        row.update(verification_thresholds=thresholds,
                   verification_metrics={metric: 0.0 for metric in metrics},
                   verification_primary_metric=primary)
        validate_subprocess_record(row, item, context, ZERO, ZERO, ZERO, None)
        for mutation in ('build', 'extra'):
            bad = copy.deepcopy(row)
            if mutation == 'build': bad['parameters']['runtime']['build_metadata_sha256'] = ZERO
            else: bad['parameters']['extra_math_parameter'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_subprocess_record(bad, item, context, ZERO, ZERO, ZERO, None)

    def test_real_dependency_file_drift_stops_before_subprocess(self):
        from gpu_suite.runner import execute_schedule, validate_manifest_artifacts
        config, _, context, schedule, _ = campaign()
        item = copy.deepcopy(schedule[0])
        entry = item['entry']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, library = root / 'fft_cpu_bench', root / 'libfftw3f.a'
            binary.write_bytes(b'not executed')
            library.write_bytes(b'approved library bytes')
            entry.update(executable_path=str(binary), binary_sha256=sha256_file(binary), supported_cpu_backends=['cpu-fftw-serial'])
            entry['artifact_id'] = artifact_id(entry)
            metadata = {key: entry[key] for key in
                        ('build_type', 'build_profile', 'git_metadata_available', 'git_commit', 'git_dirty')}
            metadata.update(vendor='amd', dependencies={'fftw3f': {'path': str(library), 'sha256': sha256_file(library)}},
                            c={key: entry[key] for key in ('compiler', 'compiler_version', 'global_configure_flags')})
            by_hash = {entry['build_metadata_sha256']: metadata}
            manifest = {'entries': [entry]}
            validate_manifest_artifacts(manifest, by_hash, 'smoke', True)
            library.write_bytes(b'different library at same path')
            with self.assertRaisesRegex(ValueError, 'dependency changed'):
                validate_manifest_artifacts(manifest, by_hash, 'smoke', True)
            output = root / 'out.jsonl'
            with mock.patch('gpu_suite.runner.subprocess.run') as launch:
                self.assertEqual(execute_schedule([item], context, output, ZERO, ZERO, by_hash, ZERO, None), 1)
                launch.assert_not_called()
            rows = load_raw_results(output)
            self.assertEqual(len(rows), item['scope_settings']['trials'])
            self.assertTrue(all(row['status'] == 'skipped' and row['failure_origin'] == 'prerequisite' for row in rows))

    def test_gpu_profiles_cannot_resolve_different_same_version_libraries(self):
        from gpu_suite.runner import validate_manifest_artifacts
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / 'librocfft.so'
            hip = root / 'libamdhip64.so'
            library.write_bytes(b'library A')
            hip.write_bytes(b'hip runtime')
            entries, metadata = [], {}
            for index, impl in enumerate(('hip', 'openmp'), 1):
                entry = amd_entry(impl)
                binary = root / entry['target_name']
                binary.write_bytes(b'not executed')
                entry.update(executable_path=str(binary), binary_sha256=sha256_file(binary),
                             build_metadata_sha256=str(index) * 64)
                entry['artifact_id'] = artifact_id(entry)
                meta = {key: entry[key] for key in
                        ('build_type', 'build_profile', 'git_metadata_available', 'git_commit', 'git_dirty')}
                meta.update(vendor='amd', dependencies={
                    'hip': {'path': str(hip), 'sha256': sha256_file(hip)},
                    'rocfft': {'path': str(library), 'sha256': sha256_file(library)}})
                meta[entry['compiler_language']] = {key: entry[key] for key in
                    ('compiler', 'compiler_version', 'global_configure_flags')}
                metadata[entry['build_metadata_sha256']] = meta
                entries.append(entry)
            validate_manifest_artifacts({'entries': entries}, metadata, 'smoke', True)
            alternate = root / 'other-librocfft.so'
            alternate.write_bytes(b'library A')
            metadata[entries[1]['build_metadata_sha256']]['dependencies']['rocfft']['path'] = str(alternate)
            with self.assertRaisesRegex(ValueError, 'identities differ'):
                validate_manifest_artifacts({'entries': entries}, metadata, 'smoke', True)

    def test_cross_vendor_campaign_is_rejected(self):
        records = campaign()[4]
        with self.assertRaises(ValueError): validate_rocfft_provenance(records + [success_record()])


if __name__ == '__main__':
    unittest.main()
