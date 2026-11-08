"""Freeze split manifests, audit overlaps and verify struct index identity."""
import argparse
from collections import Counter
from .runtime.retrieval.corpus import iter_passages, load_passages_jsonl, passage_fingerprint, write_passages_jsonl
from .common import ROOT, WORK, balanced_sample, norm, read_json, samples_from_dialogues, sha, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-limit', type=int, default=0, help='0 uses all deduplicated training inputs')
    p.add_argument('--raw-dir', type=str, default=str(ROOT / 'multidoc2dial'))
    p.add_argument('--corpus', type=str, default=str(ROOT / 'artifacts_struct_bge/passages.jsonl'))
    p.add_argument('--dev-limit', type=int, default=160)
    p.add_argument('--test-limit', type=int, default=400)
    p.add_argument('--output', type=str, default=str(WORK / 'data'))
    args = p.parse_args()
    from pathlib import Path
    out = Path(args.output)
    raw_dir = Path(args.raw_dir).resolve()
    corpus = Path(args.corpus)
    regenerated = list(iter_passages(raw_dir / 'multidoc2dial_doc.json', segmentation='struct'))
    if not corpus.exists():
        write_passages_jsonl(regenerated, corpus)
    passages = load_passages_jsonl(corpus)
    assert passage_fingerprint(passages) == passage_fingerprint(regenerated)
    all_splits, skipped = {}, {}
    files = {}
    for split in ['train', 'validation', 'test']:
        path = raw_dir / f'multidoc2dial_dial_{split}.json'
        all_splits[split], skipped[split] = samples_from_dialogues(path, passages)
        files[path.name] = sha(path)
    dialogue_ids = {s: {r['dialogue_id'] for r in rows} for s, rows in all_splits.items()}
    assert not dialogue_ids['train'] & (dialogue_ids['test'] | dialogue_ids['validation'])
    assert not dialogue_ids['test'] & dialogue_ids['validation']
    # Only query text is inspected for deduplication, never test labels or scores.
    test_queries = {norm(s['query']) for s in all_splits['test']}
    validation = [s for s in all_splits['validation'] if norm(s['query']) not in test_queries]
    heldout_queries = test_queries | {norm(s['query']) for s in all_splits['validation']}
    train = [s for s in all_splits['train'] if norm(s['query']) not in heldout_queries]
    unique_train = {}
    for row in train:
        key = norm(row['query'])
        if key not in unique_train:
            unique_train[key] = dict(row)
        else:
            unique_train[key]['gold'] = sorted(set(unique_train[key]['gold']) | set(row['gold']))
    selected = {'train': balanced_sample(list(unique_train.values()), args.train_limit),
                'dev': balanced_sample(validation, args.dev_limit),
                'test': balanced_sample(all_splits['test'], args.test_limit)}
    for split, rows in selected.items():
        write_json(out / f'{split}.json', rows)
    audit = {'seed': 20261108, 'segmentation': 'struct', 'base_model': 'BAAI/bge-base-en-v1.5',
        'query_contract': 'last_four_visible_turns_with_role_labels',
        'gold_contract': 'next_agent_referenced_spans_to_passage_ids',
        'domain_filter': False, 'corpus_count': len(passages),
        'corpus_fingerprint': passage_fingerprint(passages), 'source_sha256': files,
        'raw_evaluable': {s: len(v) for s, v in all_splits.items()}, 'skipped': skipped,
        'removed_train_queries_overlapping_heldout': len(all_splits['train']) - len(train),
        'removed_validation_queries_overlapping_test': len(all_splits['validation']) - len(validation),
        'train_duplicate_queries_merged': len(train) - len(unique_train),
        'dialogue_id_overlap': 0,
        'selection': {s: {'n': len(v), 'domain_counts': dict(Counter(r['domain'] for r in v)),
                         'sha256': sha(out / f'{s}.json')} for s, v in selected.items()},
        'document_overlap_note': 'Documents are intentionally shared across dialogue splits; this is in-domain dialogue generalization, not unseen-document evaluation.',
        'deduplication_note': 'Exact normalized full-input duplicates removed across splits. Paraphrases and short current-utterance overlap are not eliminated.'}
    write_json(out / 'audit.json', audit)
    print(audit)


if __name__ == '__main__':
    main()
