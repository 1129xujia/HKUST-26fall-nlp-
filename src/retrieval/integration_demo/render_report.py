"""Render persisted demo output as a standalone, offline HTML report."""

import argparse
from html import escape
import json
from pathlib import Path


def render(input_dir: Path, output: Path) -> None:
    cases = json.loads((input_dir / "demo_cases.json").read_text(encoding="utf-8"))
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "passed":
        raise ValueError("Only a completed, passing acceptance run can be rendered")
    e = lambda value: escape(str(value))
    rows = []
    for mode, values in summary["metrics"].items():
        rows.append(f"<tr><td>{e(mode)}</td><td>{values['hit_rate@5']:.3f}</td><td>{values['recall@5']:.3f}</td><td>{values['mrr']:.3f}</td><td>{values['mean_latency_ms']:.1f}</td></tr>")
    cards = []
    for index, case in enumerate(cases, 1):
        plan = case["plan"]
        evidence = "".join(
            f"<details><summary>{e(item['title'])} · {e(item['section_title'])}</summary><p>{e(item['text'])}</p><code>{e(item['passage_id'])}</code><p class='meta'>来源：{e(', '.join(item['source_retrievers']))} · RRF {item['rrf_score']:.4f} · 重排 {item['rerank_score']:.4f}</p></details>"
            for item in case["evidence"]
        )
        cards.append(f"""<article>
<p class="eyebrow">CASE {index:02d} · {e(plan['domain'])}</p><h2>{e(case['question'])}</h2>
<div class="flow">Part 1 查询规划 <span>→</span> Part 2 混合检索 <span>→</span> Part 3 回答与检查</div>
<h3>1 · 查询规划</h3><p>{e(plan['query'])}</p><ul>{''.join('<li>'+e(q)+'</li>' for q in plan['sub_queries'])}</ul>
<h3>2 · 证据（点击展开）</h3>{evidence}
<h3>3 · 带引用的回答</h3><blockquote>{e(case['answer']['text'])}</blockquote>
<div class="quality">引用检查：{'通过' if case['quality']['citations_valid'] else '未通过'} · 证据 {len(case['evidence'])} 条 · 无领域重试：{'是' if case['retried_without_domain'] else '否'}</div>
</article>""")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RAG 项目融合演示</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#163043;background:#f0f4f6;line-height:1.65}body{margin:0}main{max-width:1080px;margin:auto;padding:48px 24px}header{padding:24px 0 36px}h1{font-size:38px;line-height:1.2;margin:8px 0 16px}h2{font-size:23px}h3{font-size:16px;color:#246774;margin:24px 0 8px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:2px;color:#246774}article,.overview{background:white;border:1px solid #dbe4e8;border-radius:16px;padding:28px;margin:20px 0;box-shadow:0 8px 32px #16304306}.flow{display:flex;flex-wrap:wrap;gap:12px;background:#edf5f6;padding:14px;border-radius:8px;font-size:14px}.flow span{color:#69939c}details{border-bottom:1px solid #e2e9ed;padding:12px 0}summary{cursor:pointer;font-weight:600}code{overflow-wrap:anywhere;font-size:11px;background:#eef3f5;padding:4px 6px}.meta{font-size:12px;color:#667d8a}blockquote{margin:0;padding:20px;background:#f4f7f8;border-left:3px solid #2a7d80;overflow-wrap:anywhere}.quality{color:#236e58;font-size:13px;margin-top:16px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e9ed}.note{color:#627784;font-size:14px}.scroll{overflow-x:auto}a{color:#246774}</style><main>
"""
    page += f"<header><p class='eyebrow'>HKUST · MULTI-AGENT RAG</p><h1>从用户问题到可追溯回答</h1><p>独立 Part 1 / Part 3 Demo，与正式 Part 2 检索模块通过数据契约连接。</p><p class='note'>真实语料 {summary['corpus_passages']:,} 段 · {e('完整模型验收' if summary['full'] else 'BM25 快速验收')} · {e(summary['finished_at_utc'])}</p></header>"
    page += "<section class='overview'><h2>验证集检索对比</h2><div class='scroll'><table><tr><th>模式</th><th>Hit@5</th><th>Recall@5</th><th>MRR@10</th><th>平均延迟 ms</th></tr>" + "".join(rows) + "</table></div><p class='note'>四领域均衡小样本验收，仅用于验证集成与观察趋势。延迟不含模型加载；MRR 只在返回的前 10 条中计算。</p></section>"
    page += "".join(cards)
    page += "<footer class='note'>Part 1 使用规则规划；Part 3 使用抽取式回答。引用检查验证引用对应关系，不能代替语义事实核查。此报告完全离线，无外部脚本或服务依赖。</footer></main></html>"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/acceptance"))
    parser.add_argument("--output", type=Path, default=Path("integration_demo/output/demo_report.html"))
    args = parser.parse_args()
    render(args.input_dir, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
