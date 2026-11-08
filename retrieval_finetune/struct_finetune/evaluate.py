"""Evaluate frozen or adapted embeddings with shared BM25/RRF/CE components."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

from .common import PREFIX, ROOT, WORK, metrics, read_json, sha, summarize, write_json
os.environ.setdefault('HF_HOME', str(ROOT / '.model_cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--corpus', type=Path, default=ROOT / 'artifacts_struct_bge/passages.jsonl')
    parser.add_argument('--samples', type=Path, default=WORK / 'data/dev.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--reuse-vectors', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--reranker-model', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    parser.add_argument('--segmentation', choices=['struct', 'section'], default='struct')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    import faiss
    import numpy as np
    from .runtime.retrieval.bm25 import BM25Retriever
    from .runtime.retrieval.corpus import load_passages_jsonl, passage_fingerprint, PASSAGE_FINGERPRINT_VERSION
    from .runtime.retrieval.fusion import reciprocal_rank_fusion
    from .runtime.retrieval.reranker import CrossEncoderReranker
    from .runtime.retrieval.schemas import SearchHit
    from .train import load_model
    passages = load_passages_jsonl(args.corpus)
    samples = read_json(args.samples)
    if not samples: raise ValueError('No samples')
    ids = {p.passage_id for p in passages}
    assert all(set(s['gold']) <= ids for s in samples)
    model = load_model(args.model, args.device)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(args.corpus, args.index_dir / 'passages.jsonl')
    write_json(args.index_dir / 'corpus_metadata.json', {'segmentation': args.segmentation,
        'passage_count': len(passages), 'gold_contract': 'next_agent_referenced_spans_to_passage_ids',
        'passage_fingerprint': passage_fingerprint(passages)})
    begin = time.perf_counter()
    identity = {'model_name': str(args.model), 'corpus_sha256': sha(args.corpus),
                'query_prefix': PREFIX, 'passage_prefix': '', 'max_seq_length': model.max_seq_length}
    for filename in ['model.safetensors', 'pytorch_model.bin']:
        candidate = Path(args.model) / filename
        if candidate.is_file(): identity['model_weight_sha256'] = sha(candidate)
    vectors_path = args.index_dir / 'passage_embeddings.npy'
    if vectors_path.is_file() and read_json(args.index_dir / 'identity.json') == identity:
        vectors = np.load(vectors_path)
        index_mode = 'cached_identical_model_and_corpus'
    elif args.reuse_vectors:
        source_meta = read_json(args.reuse_vectors / 'dense_metadata.json')
        assert source_meta['model_name'] == args.model
        assert source_meta['passage_fingerprint'] == passage_fingerprint(passages)
        assert source_meta.get('passage_prefix', '') == ''
        vectors = np.load(args.reuse_vectors / 'passage_embeddings.npy')
        index_mode = 'existing_frozen_base_vectors'
    else:
        vectors = model.encode([p.index_text for p in passages], batch_size=args.batch_size,
            normalize_embeddings=True, show_progress_bar=True).astype('float32')
        index_mode = 'fresh_full_corpus_encoding'
    assert vectors.shape == (len(passages), 768) and np.isfinite(vectors).all()
    index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
    np.save(vectors_path, vectors); faiss.write_index(index, str(args.index_dir / 'faiss.index'))
    write_json(args.index_dir / 'identity.json', identity)
    write_json(args.index_dir / 'dense_metadata.json', {'model_name': args.model,
        'query_prefix': PREFIX, 'passage_prefix': '', 'passage_count': len(passages),
        'dimension': 768, 'normalized': True, 'fingerprint_version': PASSAGE_FINGERPRINT_VERSION,
        'passage_fingerprint': passage_fingerprint(passages), 'segmentation': args.segmentation,
        'backend': 'sentence_transformer'})
    index_elapsed = time.perf_counter() - begin
    bm25 = BM25Retriever.build(passages)
    bm25.save(args.index_dir)
    reranker = CrossEncoderReranker(args.reranker_model, device=args.device, batch_size=args.batch_size,
        query_strategy='current_first_user_history', use_sub_queries=False,
        skip_acknowledgements=True) if args.rerank else None
    # Match all CPU arms; no query batching or warmup included in timings.
    torch.set_num_threads(1)
    model.encode([PREFIX + samples[0]['query']], normalize_embeddings=True)
    groups = {'dense': [], 'rrf': []}
    if reranker: groups['full'] = []
    start = time.perf_counter()
    checkpoint_file = args.output.with_suffix('.progress.json')
    for number, sample in enumerate(samples, 1):
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        tick = time.perf_counter()
        v = model.encode([PREFIX + sample['query']], normalize_embeddings=True).astype('float32')
        if not np.isfinite(v).all(): raise RuntimeError('Invalid query embedding')
        scores, rows = index.search(v, 200)
        dense = [SearchHit(passages[int(i)], 'dense', n + 1, float(s), sample['query'])
                 for n, (i, s) in enumerate(zip(rows[0], scores[0]))]
        dense_ms = (time.perf_counter() - tick) * 1000
        tick = time.perf_counter()
        lexical = bm25.search(sample['query'], top_k=50)
        fused = reciprocal_rank_fusion([dense[:100], lexical], rrf_k=30)[:50]
        hybrid_ms = dense_ms + (time.perf_counter() - tick) * 1000
        rankings = {'dense': ([h.passage.passage_id for h in dense], dense_ms),
                    'rrf': ([h.passage.passage_id for h in fused], hybrid_ms)}
        if reranker:
            tick = time.perf_counter()
            reranked = reranker.rerank(sample['query'], [], deepcopy(fused))
            rankings['full'] = ([h.passage.passage_id for h in reranked],
                               hybrid_ms + (time.perf_counter() - tick) * 1000)
        lookup = {p.passage_id: p for p in passages}
        evidence = {tuple(x) for x in sample['evidence_spans']}
        for mode, (ranking, latency) in rankings.items():
            top_spans = {(lookup[pid].doc_id, sid) for pid in ranking[:10] for sid in lookup[pid].span_ids}
            stats = metrics(ranking, sample['gold'])
            stats['span_recall@10'] = len(top_spans & evidence) / len(evidence)
            stats['span_hit@10'] = float(bool(top_spans & evidence))
            groups[mode].append({'sample_id': sample['sample_id'], 'domain': sample['domain'],
                'gold': sample['gold'], 'ranking': ranking, 'metrics': stats, 'latency_ms': latency})
        if number == 1 or number % 10 == 0:
            print(f'Evaluated {number}/{len(samples)} elapsed={time.perf_counter()-start:.1f}s', flush=True)
            write_json(checkpoint_file, {'completed': number, 'total': len(samples)})
    payload = {'configuration': {**vars(args), 'model': str(args.model),
        'samples_sha256': sha(args.samples), 'query_prefix': PREFIX, 'pooling': 'CLS',
        'max_seq_length': model.max_seq_length, 'normalize_embeddings': True,
        'corpus_fingerprint': passage_fingerprint(passages), 'corpus_count': len(passages),
        'dense_top_k': 100, 'bm25_top_k': 50, 'rrf_k': 30, 'rerank_candidates': 50,
        'query_contract': 'last_four_visible_turns_with_role_labels',
        'gold_contract': 'next_agent_referenced_spans_to_passage_ids', 'domain_filter': False,
        'extra_dedup': False, 'timing_note': 'Warm serial query timings; shared dense and lexical timings reused across arms; excludes model load and index construction.',
        'index_mode': index_mode, 'index_elapsed_s': index_elapsed}, 'results': {}}
    payload['configuration'] = {k: str(v) if isinstance(v, Path) else v for k,v in payload['configuration'].items()}
    for name, rows in groups.items():
        payload['results'][name] = {'summary': summarize(rows), 'by_domain': {
            d: summarize([r for r in rows if r['domain'] == d]) for d in sorted({r['domain'] for r in rows})}, 'samples': rows}
    write_json(args.output, payload)
    print({name: result['summary'] for name, result in payload['results'].items()}, flush=True)


if __name__ == '__main__':
    main()
