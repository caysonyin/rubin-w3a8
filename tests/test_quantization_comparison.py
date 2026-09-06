import copy
import json
import math
import pytest
from scripts.compare_quantization import compare, ROWS
from src.experiment import FORMAT, CONFIG


def fixtures():
    rows = []
    for index, (mode, method) in enumerate(ROWS):
        rows.append(dict(spec_version='reference-v1', experiment_version='quantization-comparison-v1',
            model_path='local', device='cpu', dataset='Salesforce/wikitext', dataset_config='wikitext-2-raw-v1',
            split='test', seq_len=1024, num_eval_tokens=32768, num_predicted_tokens=32736, seed=42,
            mode=mode, weight_method=method, mean_nll=2 + index * .1, ppl=math.exp(2 + index * .1),
            model_fingerprint='a'*64, tokenizer_fingerprint='b'*64, eval_token_hash='c'*64,
            quantization_format=FORMAT, recipe_config=CONFIG, python_version='3.14', torch_version='test',
            transformers_version='test', cpu_threads=20, full_model_smoke='PASS',
            payload_fingerprint=None if index == 0 else ('d' if index < 3 else 'e')*64,
            calibration=None if index < 3 else dict(dataset='Salesforce/wikitext', config='wikitext-2-raw-v1',
                split='train', tokens=512, token_hash='f'*64, source='unquantized_bf16_inputs_fp32_rotation')))
    return rows


def write(tmp_path, rows):
    paths = [tmp_path / f'{i}.json' for i in range(5)]
    for path, row in zip(paths, rows): path.write_text(json.dumps(row))
    return paths


def test_comparison_deltas(tmp_path):
    summary, contrasts = compare(write(tmp_path, fixtures()))
    assert len(summary) == 5 and len(contrasts) == 4
    assert contrasts[0]['delta_mean_nll'] == pytest.approx(.2)
    assert contrasts[2]['delta_mean_nll'] == pytest.approx(.1)


@pytest.mark.parametrize('field,value', [('eval_token_hash', 'different'), ('payload_fingerprint', 'different'),
    ('mean_nll', float('nan')), ('ppl', 3), ('weight_method', 'none'), ('full_model_smoke', 'FAIL'),
    ('calibration', {'split': 'test'}), ('recipe_config', {}), ('seq_len', 512)])
def test_comparison_refuses_invalid_rows(tmp_path, field, value):
    rows = copy.deepcopy(fixtures())
    rows[4][field] = value
    with pytest.raises(ValueError): compare(write(tmp_path, rows))


def test_missing_metadata_refused(tmp_path):
    rows = fixtures()
    del rows[2]['model_fingerprint']
    with pytest.raises(ValueError): compare(write(tmp_path, rows))


def test_empty_common_fingerprint_refused(tmp_path):
    rows = fixtures()
    for row in rows: row['model_fingerprint'] = ''
    with pytest.raises(ValueError): compare(write(tmp_path, rows))


def test_runner_prepares_once_and_reuses_payloads(tmp_path, monkeypatch):
    from scripts import run_quantization_comparison as runner
    rows = fixtures()
    historical = tmp_path / 'results'
    historical.mkdir()
    for name, row in zip(('bf16', 'w3a16', 'w3a8'), rows):
        (historical / f'{name}.json').write_text(json.dumps(row))
    output = tmp_path / 'comparison'
    calls = []
    def execute(command, *, env, **kwargs):
        calls.append((command, env))
        if command[2] == 'scripts.compare_quantization':
            compare(command[3:8])
            return
        def flag(name): return command[command.index(name) + 1]
        if '--prepare-only' in command:
            assert env['OMP_NUM_THREADS'] == '4'
            from pathlib import Path
            Path(flag('--save-payload')).write_text('test payload')
        else:
            assert env['OMP_NUM_THREADS'] == '20'
            mode, method = flag('--mode'), flag('--weight-method')
            expected_method = 'none' if mode == 'bf16' else method
            if mode != 'bf16':
                from pathlib import Path
                assert Path(flag('--load-payload')).is_file()
                assert '--save-payload' not in command
            row = rows[ROWS.index((mode, expected_method))]
            from pathlib import Path
            Path(flag('--output')).write_text(json.dumps(row))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    monkeypatch.setattr(runner.sys, 'argv', ['runner', '--output-dir', str(output)])
    runner.main()
    assert sum('--prepare-only' in command for command, _ in calls) == 2
    assert sum(command[2] == 'scripts.eval_qwen' and '--prepare-only' not in command for command, _ in calls) == 5
    calls.clear()
    monkeypatch.setattr(runner.sys, 'argv', ['runner', '--output-dir', str(output), '--reuse-payloads', '--resume-evaluations'])
    runner.main()
    assert len(calls) == 1 and calls[0][0][2] == 'scripts.compare_quantization'
    row = json.loads((output / 'bf16.json').read_text())
    row['cpu_threads'] = 4
    (output / 'bf16.json').write_text(json.dumps(row))
    with pytest.raises(ValueError): runner.main()
