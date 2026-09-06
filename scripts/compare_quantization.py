"""Validate and compare the five frozen quantization experiment rows."""
import argparse
import json
from pathlib import Path
from scripts.compare_results import _load_and_validate, _validate_metadata_match, _write_csv
from src.experiment import METHODS, FORMAT, CONFIG

ROWS = [('bf16', 'none'), ('w3a16', METHODS[0]), ('w3a8', METHODS[0]),
        ('w3a16', METHODS[1]), ('w3a8', METHODS[1])]
EXTRA = ('experiment_version', 'model_fingerprint', 'tokenizer_fingerprint', 'eval_token_hash',
         'quantization_format', 'recipe_config', 'python_version', 'torch_version',
         'transformers_version', 'cpu_threads')


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def compare(paths):
    if len(paths) != 5:
        raise ValueError('Exactly five result files required in frozen row order')
    results = [_load_and_validate(Path(p), mode) for p, (mode, _) in zip(paths, ROWS)]
    _validate_metadata_match([(str(p), r) for p, r in zip(paths, results)])
    for r, (mode, method) in zip(results, ROWS):
        for key in EXTRA:
            if key not in r or r[key] != results[0].get(key):
                raise ValueError(f'Missing/mismatched {key}')
        for key in ('model_fingerprint', 'tokenizer_fingerprint', 'eval_token_hash'):
            if not is_sha256(r[key]):
                raise ValueError(f'Invalid {key}')
        if mode != 'bf16' and not is_sha256(r.get('payload_fingerprint')):
            raise ValueError('Invalid payload fingerprint')
        if r.get('weight_method') != method or r.get('full_model_smoke') != 'PASS':
            raise ValueError('Recipe or smoke mismatch')
        if r['quantization_format'] != FORMAT or r['recipe_config'] != CONFIG:
            raise ValueError('Unexpected quantization configuration')
        if method == METHODS[1]:
            c = r.get('calibration')
            if not isinstance(c, dict) or c.get('split') != 'train' or c.get('tokens') != 512 or c.get('dataset') != 'Salesforce/wikitext' or c.get('config') != 'wikitext-2-raw-v1' or c.get('source') != 'unquantized_bf16_inputs_fp32_rotation' or not is_sha256(c.get('token_hash')):
                raise ValueError('Invalid train calibration metadata')
        elif r.get('calibration') is not None:
            raise ValueError('Unexpected calibration')
        if mode == 'bf16' and r.get('payload_fingerprint') is not None:
            raise ValueError('BF16 cannot have a payload')
    if results[0]['experiment_version'] != 'quantization-comparison-v1':
        raise ValueError('Unexpected experiment version')
    for a, b in ((1, 2), (3, 4)):
        if not results[a].get('payload_fingerprint') or results[a]['payload_fingerprint'] != results[b].get('payload_fingerprint') or results[a]['calibration'] != results[b]['calibration']:
            raise ValueError('Paired weight payload or calibration mismatch')
    canonical = dict(cpu_threads=20, device='cpu', dataset='Salesforce/wikitext', dataset_config='wikitext-2-raw-v1', split='test', seq_len=1024, num_eval_tokens=32768, num_predicted_tokens=32736, seed=42)
    if any(results[0].get(k) != v for k, v in canonical.items()):
        raise ValueError('Noncanonical evaluation protocol')
    summary = []
    for r in results:
        summary.append(dict(weight_method=r['weight_method'], mode=r['mode'], mean_nll=r['mean_nll'], ppl=r['ppl'],
                            delta_mean_nll=r['mean_nll'] - results[0]['mean_nll'], delta_ppl=r['ppl'] - results[0]['ppl']))
    contrasts = []
    for name, a, b in [('recipe_w3a16', 1, 3), ('recipe_w3a8', 2, 4), ('activation_plain', 1, 2), ('activation_h64_gptq_refit', 3, 4)]:
        contrasts.append(dict(contrast=name, delta_mean_nll=results[b]['mean_nll'] - results[a]['mean_nll'], delta_ppl=results[b]['ppl'] - results[a]['ppl']))
    return summary, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs=5, type=Path, help='BF16, plain W3A16/W3A8, optimized W3A16/W3A8')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    summary, contrasts = compare(args.results)
    _write_csv(args.output_dir / 'summary.csv', summary)
    _write_csv(args.output_dir / 'contrasts.csv', contrasts)
    print(json.dumps({'summary': summary, 'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
