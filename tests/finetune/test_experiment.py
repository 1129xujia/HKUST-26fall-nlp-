import math
import pytest
import torch
from experiments.finetune.common import metrics, samples_from_dialogues, write_json
from experiments.finetune.train import contrastive_loss


def test_metrics_multi_gold_and_cutoff():
    result = metrics(['wrong', 'g1'] + [f'n{i}' for i in range(8)] + ['g2'], ['g1','g2'])
    assert result['hit@10'] == 1
    assert result['recall@10'] == .5
    assert result['mrr@10'] == .5
    assert result['ndcg@10'] == pytest.approx((1/math.log2(3))/(1+1/math.log2(3)))
    assert result['recall@50'] == 1
    with pytest.raises(ValueError): metrics(['x','x'],['x'])
    with pytest.raises(ValueError): metrics(['x'],[])


def test_future_answer_never_enters_query(tmp_path):
    from customer_service_rag.retrieval.schemas import Passage
    p = Passage('doc::structure_0','doc','dmv','title','','','text',['1'])
    path=tmp_path/'dialogues.json'
    write_json(path, {'dial_data':{'dmv':[{'dial_id':'dialogue', 'turns':[
        {'role':'user','turn_id':1,'utterance':'question'},
        {'role':'agent','turn_id':2,'utterance':'FUTURE_SECRET','da':'respond_solution',
         'references':[{'doc_id':'doc','id_sp':'1'}]}]}]}})
    rows,skipped=samples_from_dialogues(path,[p])
    assert rows[0]['query']=='User: question'
    assert 'FUTURE_SECRET' not in rows[0]['query']
    assert rows[0]['gold']==['doc::structure_0']


def test_masked_multi_positive_loss_and_gradient():
    q=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    d=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],requires_grad=True)
    positive=torch.tensor([[True,True,False],[False,False,True]])
    allowed=torch.ones_like(positive)
    loss=contrastive_loss(q,d,positive,allowed,scale=1.)
    expected=((torch.logsumexp(torch.tensor([1.,1.,0.]),0)-math.log(2)-1)
             +(torch.logsumexp(torch.tensor([0.,0.,1.]),0)-1))/2
    assert loss.item()==pytest.approx(expected.item())
    loss.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(d.grad).all()
