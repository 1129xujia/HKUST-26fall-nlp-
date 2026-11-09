"""Budgeted partial BGE fine-tuning with multi-positive masked contrastive loss."""
import argparse
import json
import os
from pathlib import Path
import random
import time

from .common import PREFIX, ROOT, WORK, SEED, read_json, sha, write_json
os.environ.setdefault('HF_HOME', str(ROOT / '.model_cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def contrastive_loss(q, d, positive, allowed, scale=20.):
    import torch
    scores = torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(d, dim=-1).T * scale
    return (torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), 1)
            - torch.logsumexp(scores.masked_fill(~positive, -torch.inf), 1)).mean()


def load_model(path, device='cpu'):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(path), device=device)
    # Own storage for this macOS/Torch combination; numerical values unchanged.
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data.clone()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=WORK / 'data')
    p.add_argument('--output', type=Path, default=WORK / 'run')
    p.add_argument('--steps', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--layers', type=int, default=2, help='Number of final BERT layers to train; 12 trains all encoder layers')
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--base-model', default='BAAI/bge-base-en-v1.5')
    p.add_argument('--corpus', type=Path, default=ROOT / 'artifacts/indexes/finetune/passages.jsonl')
    p.add_argument('--checkpoint-every', type=int, default=256)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 2 or not 1 <= args.layers <= 12:
        p.error('steps>=1, batch-size>=2, layers in [1,12] required')
    import numpy as np
    import torch
    from customer_service_rag.retrieval.corpus import load_passages_jsonl
    torch.set_num_threads(args.threads)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    rows = read_json(args.data / 'train.json')
    passages = load_passages_jsonl(args.corpus)
    lookup = {x.passage_id: x for x in passages}
    model = load_model(args.base_model, args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in model[0].auto_model.encoder.layer[-args.layers:]:
        for param in layer.parameters():
            param.requires_grad_(True)
    if args.layers == 12:
        for param in model.parameters():
            param.requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    rng = random.Random(SEED)
    rng.shuffle(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(seed=SEED, base_model=args.base_model, segmentation='struct',
        max_seq_length=model.max_seq_length, pooling=model[1].get_config_dict(),
        query_prefix=PREFIX, passage_prefix='', normalize_embeddings=True,
        loss='multi_positive_logsumexp_scale20_same_document_unlabelled_mask',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=sum(p.numel() for p in model.parameters()),
        train_sha256=sha(args.data / 'train.json'),
        false_negative_policy='Union all labelled positives per normalized training input; mask unlabelled positives from the same document. Other documents are presumed negatives, not guaranteed irrelevant.')
    write_json(args.output / 'training_config.json', config)
    start = time.perf_counter()
    history, seen = [], set()
    for step in range(1, args.steps + 1):
        # Build a batch with distinct documents so masking leaves useful negatives.
        batch, doc_ids = [], set()
        attempts = 0
        while len(batch) < args.batch_size:
            item = rows[rng.randrange(len(rows))]
            docs = {lookup[g].doc_id for g in item['gold']}
            attempts += 1
            if docs & doc_ids:
                if attempts > 10000: raise RuntimeError('Not enough distinct positive documents')
                continue
            batch.append(item); doc_ids |= docs; seen.add(item['sample_id'])
        ids = sorted({g for s in batch for g in s['gold']})
        positive = torch.tensor([[pid in s['gold'] for pid in ids] for s in batch])
        allowed = torch.tensor([[pid in s['gold'] or lookup[pid].doc_id not in
            {lookup[g].doc_id for g in s['gold']} for pid in ids] for s in batch])
        positive = positive.to(model.device)
        allowed = allowed.to(model.device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        def features(texts):
            return {k: v.to(model.device) for k, v in model.tokenize(texts).items()}
        q = model(features([PREFIX + s['query'] for s in batch]))['sentence_embedding']
        d = model(features([lookup[pid].index_text for pid in ids]))['sentence_embedding']
        loss = contrastive_loss(q, d, positive, allowed)
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        warm = min(1., step / max(1, args.steps * .1))
        decay = min(1., (args.steps - step + 1) / max(1., args.steps * .9))
        for group in optimizer.param_groups: group['lr'] = args.lr * warm * decay
        optimizer.step()
        history.append({'step': step, 'loss': loss.item(), 'elapsed_s': time.perf_counter() - start})
        if step == 1 or step % 10 == 0:
            print(json.dumps(history[-1]), flush=True)
            write_json(args.output / 'loss_history.json', history)
        if step % args.checkpoint_every == 0 or step == args.steps:
            model.eval()
            checkpoint = args.output / f'checkpoint-{step}'
            model.save(str(checkpoint))
            write_json(checkpoint / 'experiment_provenance.json', {**config, 'step': step})
    model.eval()
    # Reload and compare independently owned saved weights.
    texts = [PREFIX + rows[0]['query'], lookup[rows[0]['gold'][0]].index_text]
    before = model.encode(texts, normalize_embeddings=True)
    restored = load_model(args.output / f'checkpoint-{args.steps}', args.device)
    after = restored.encode(texts, normalize_embeddings=True)
    error = float(np.max(np.abs(before - after)))
    if not np.isfinite(after).all() or error > 1e-4: raise RuntimeError('Reload parity failed')
    write_json(args.output / 'training_summary.json', {'elapsed_s': time.perf_counter() - start,
        'steps': args.steps, 'unique_queries_seen': len(seen), 'available_train_queries': len(rows),
        'examples_drawn': args.steps * args.batch_size, 'reload_max_abs_error': error,
        'first_loss': history[0]['loss'], 'last_loss': history[-1]['loss'],
        'mean_last_20_loss': float(np.mean([h['loss'] for h in history[-20:]])),
        'model_bytes': sum(f.stat().st_size for f in (args.output / f'checkpoint-{args.steps}').rglob('*') if f.is_file())})
    write_json(args.output / 'loss_history.json', history)


if __name__ == '__main__':
    main()
