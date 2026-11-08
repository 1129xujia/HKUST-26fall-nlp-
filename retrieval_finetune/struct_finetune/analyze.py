"""Validation-only checkpoint selection and paired final comparisons."""
import argparse
from pathlib import Path
from .common import SEED, read_json, write_json


def select(paths, output):
    candidates = []
    for path in paths:
        result = read_json(path)
        stats = result['results']['dense']['summary']
        candidates.append({'model': result['configuration']['model'], 'result_file': str(path),
                           'summary': stats})
    chosen = max(candidates, key=lambda c: (c['summary']['hit@10'], c['summary']['mrr@10'], c['summary']['ndcg@10']))
    write_json(output, {'selection_split': 'dev_only', 'criterion': 'Hit@10, then MRR@10, then nDCG@10',
                        'selected': chosen, 'candidates': candidates})
    print(chosen['model'])


def paired(left, right, samples, iterations=2000):
    import numpy as np
    from scipy.stats import binomtest
    assert [r['sample_id'] for r in left] == [r['sample_id'] for r in right]
    assert all(a['gold'] == b['gold'] for a,b in zip(left, right))
    dialogue = {s['sample_id']: s['dialogue_id'] for s in samples}
    ids = sorted({dialogue[r['sample_id']] for r in left})
    groups = {d: [] for d in ids}
    for a,b in zip(left, right):
        groups[dialogue[a['sample_id']]].append(a['metrics']['hit@10']-b['metrics']['hit@10'])
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(iterations):
        draws = rng.choice(ids, len(ids), replace=True)
        values = [v for d in draws for v in groups[d]]
        boot.append(np.mean(values))
    wins = sum(a['metrics']['hit@10'] > b['metrics']['hit@10'] for a,b in zip(left,right))
    losses = sum(a['metrics']['hit@10'] < b['metrics']['hit@10'] for a,b in zip(left,right))
    return {'delta_hit10': float(np.mean([a['metrics']['hit@10']-b['metrics']['hit@10'] for a,b in zip(left,right)])),
            'dialogue_cluster_bootstrap_95ci': np.percentile(boot,[2.5,97.5]).tolist(),
            'wins': wins, 'losses': losses, 'ties': len(left)-wins-losses,
            'mcnemar_exact_query_level_p': binomtest(wins, wins+losses).pvalue if wins+losses else 1.,
            'inference_note': 'Primary CI resamples dialogue clusters. Exact McNemar is supplemental and assumes independent queries.'}


def compare(base_path, tuned_path, samples_path, output):
    base, tuned, samples = read_json(base_path), read_json(tuned_path), read_json(samples_path)
    assert base['configuration']['samples_sha256'] == tuned['configuration']['samples_sha256']
    assert base['configuration']['corpus_fingerprint'] == tuned['configuration']['corpus_fingerprint']
    arms = {'A': base['results']['dense'], 'B': tuned['results']['dense'],
            'C': base['results']['rrf'], 'D': base['results']['full']}
    result = {'arms': {k: {'summary': v['summary'], 'by_domain': v['by_domain']} for k,v in arms.items()},
              'comparisons': {f'{a}_minus_{b}': paired(arms[a]['samples'], arms[b]['samples'], samples)
                              for a,b in [('B','D'),('B','A'),('D','C')]},
              'evidence': {'base': str(base_path), 'tuned': str(tuned_path), 'samples': str(samples_path)}}
    write_json(output, result)
    table = ['# Struct final comparison', '', '| Arm | Hit@10 | Recall@10 | MRR@10 | nDCG@10 | Mean ms |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, values in result['arms'].items():
        s=values['summary']
        table.append(f"| {arm} | {s['hit@10']:.2%} | {s['recall@10']:.2%} | {s['mrr@10']:.4f} | {s['ndcg@10']:.4f} | {s['latency_mean_ms']:.1f} |")
    table += ['', 'A frozen Dense; B adapted Dense; C frozen Dense + BM25 + RRF; D C + Cross-Encoder.',
              '', '## Paired comparisons', '', '```json', __import__('json').dumps(result['comparisons'], indent=2), '```']
    Path(output).with_suffix('.md').write_text('\n'.join(table)+'\n')
    queries={s['sample_id']:s['query'] for s in samples}
    failure=[]
    for b,d in zip(arms['B']['samples'],arms['D']['samples']):
        if b['metrics']['hit@10'] != d['metrics']['hit@10']:
            failure.append({'sample_id':b['sample_id'],'domain':b['domain'],'query':queries[b['sample_id']],
                'gold':b['gold'],'B_hit':b['metrics']['hit@10'],'D_hit':d['metrics']['hit@10'],
                'B_top10':b['ranking'][:10],'D_top10':d['ranking'][:10]})
    write_json(Path(output).with_name('discordant_cases.json'), failure)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('select');s.add_argument('paths',nargs='+',type=Path);s.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('compare');c.add_argument('--base',type=Path,required=True);c.add_argument('--tuned',type=Path,required=True)
    c.add_argument('--samples',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='select':select(a.paths,a.output)
    else:compare(a.base,a.tuned,a.samples,a.output)


if __name__=='__main__':main()
