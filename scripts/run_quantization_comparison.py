"""Run five CPU evaluations with one saved payload per recipe; preserve old results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='models/Qwen3-0.6B-Base')
    parser.add_argument('--output-dir', type=Path, default=Path('results/quantization_comparison'))
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--preparation-threads', type=int, default=4)
    parser.add_argument('--reuse-payloads', action='store_true', help='Validate and reuse existing local payloads')
    parser.add_argument('--resume-evaluations', action='store_true', help='Reuse completed rows; validate all five before summary')
    parser.add_argument('--skip-bf16', action='store_true', help='Use an already completed BF16 row in this output directory')
    args = parser.parse_args()
    if args.threads < 1 or args.preparation_threads < 1:
        raise ValueError('threads must be positive')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads),
           'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
    rows = [('bf16', 'plain_lloyd_max', 'bf16'),
            ('w3a16', 'plain_lloyd_max', 'plain_w3a16'),
            ('w3a8', 'plain_lloyd_max', 'plain_w3a8'),
            ('w3a16', 'h64_gptq_refit', 'h64_gptq_refit_w3a16'),
            ('w3a8', 'h64_gptq_refit', 'h64_gptq_refit_w3a8')]
    for method in ('plain_lloyd_max', 'h64_gptq_refit'):
        if args.reuse_payloads:
            if not (out / f'{method}.pt').is_file():
                raise ValueError(f'Missing reusable payload: {method}')
            continue
        command = [sys.executable, '-m', 'scripts.eval_qwen', '--mode', 'w3a16',
                   '--weight-method', method, '--model-path', args.model_path,
                   '--prepare-only', '--save-payload', str(out / f'{method}.pt'),
                   '--output', str(out / f'prepare_{method}.json')]
        prep_env = {**env, 'OMP_NUM_THREADS': str(args.preparation_threads),
                    'MKL_NUM_THREADS': str(args.preparation_threads)}
        print(f'Preparing {method}', flush=True)
        with (out / f'prepare_{method}.log').open('w') as log:
            subprocess.run(command, env=prep_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for mode, method, name in rows:
        if args.resume_evaluations and (out / f'{name}.json').is_file():
            existing = json.loads((out / f'{name}.json').read_text())
            expected_method = 'none' if mode == 'bf16' else method
            if existing.get('cpu_threads') != args.threads or existing.get('mode') != mode or existing.get('weight_method') != expected_method:
                raise ValueError(f'Incompatible completed row: {name}')
            print(f'Reusing completed {name}', flush=True)
            continue
        if mode == 'bf16' and args.skip_bf16:
            if not (out / 'bf16.json').is_file():
                raise ValueError('No completed BF16 row to skip')
            continue
        command = [sys.executable, '-m', 'scripts.eval_qwen', '--mode', mode,
                   '--weight-method', method, '--model-path', args.model_path,
                   '--output', str(out / f'{name}.json')]
        if mode != 'bf16':
            command += ['--load-payload', str(out / f'{method}.pt')]
        print(f'Running {name}', flush=True)
        with (out / f'{name}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [sys.executable, '-m', 'scripts.compare_quantization',
               *[str(out / f'{name}.json') for _, _, name in rows], '--output-dir', str(out)]
    with (out / 'comparison.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    checks = {}
    for old, new in [('bf16', 'bf16'), ('w3a16', 'plain_w3a16'), ('w3a8', 'plain_w3a8')]:
        historical = json.loads((Path('results') / f'{old}.json').read_text())
        current = json.loads((out / f'{new}.json').read_text())
        dn = current['mean_nll'] - historical['mean_nll']
        dp = current['ppl'] - historical['ppl']
        checks[old] = {'delta_mean_nll': dn, 'delta_ppl': dp, 'pass': abs(dn) <= 1e-5 and abs(dp) <= 1e-3}
    (out / 'baseline_reproduction.json').write_text(json.dumps(checks, indent=2) + '\n')
    if not all(check['pass'] for check in checks.values()):
        raise ValueError('Historical baseline drift: investigate before accepting results')
    print('Five-row comparison and baseline reproduction: PASS', flush=True)


if __name__ == '__main__':
    main()
