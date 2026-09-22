"""Additional seeds/shapes and authentic FLA-generated selections.

Q/K/V are synthetic. The nsa pattern uses upstream mean pooling, compression
attention, and top-k kernels; it is not a trained model's activation trace.
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
import torch
import compare_baselines as harness
from experiments import Experiments

BASE_INPUTS = harness.inputs
ENGINE = None


def cases(suite):
    result = []
    if suite == 'model-shapes':
        return [dict(name=f'modelshape_n{n}_s29', lengths=[n], group=16, kv_heads=2,
                     head_dim=64, block_size=16, selected_blocks=16,
                     pattern='nsa', seed=29, sampled=True) for n in (2048, 8192)]
    if suite in ('all', 'edges'):
        specs = [
            dict(lengths=[129], head_dim=64, selected_blocks=1),
            dict(lengths=[193], head_dim=96, value_dim=80, selected_blocks=4),
            dict(lengths=[257], head_dim=256, value_dim=128, selected_blocks=2),
            dict(lengths=[129], head_dim=128, value_dim=256, selected_blocks=2),
            dict(lengths=[31, 33, 65], head_dim=64, block_size=32, selected_blocks=2),
            dict(lengths=[127, 129], block_size=128, selected_blocks=2),
            dict(lengths=[129], group=32, kv_heads=2, selected_blocks=4),
            dict(lengths=[257], dtype='fp16', selected_blocks=8),
        ]
        for seed in (3, 29, 101):
            for i, spec in enumerate(specs):
                c = dict(name=f'edge{i}_s{seed}', lengths=[129], group=16, kv_heads=1,
                         head_dim=128, block_size=64, selected_blocks=2, pattern='random', seed=seed)
                c.update(spec)
                result.append(c)
    if suite in ('all', 'matrix'):
        for n in (4096, 8192, 16384):
            for pattern in ('random', 'recent', 'nsa'):
                result.append(dict(name=f'n{n}_{pattern}_s29', lengths=[n], group=16, kv_heads=4,
                                   selected_blocks=16, pattern=pattern, seed=29, sampled=True))
    return result


def study_inputs(case):
    q, k, v, idx, rows, digest = BASE_INPUTS({**case, 'pattern': 'random' if case['pattern'] == 'nsa' else case['pattern']})
    if case['pattern'] != 'nsa':
        return q, k, v, idx, rows, digest
    assert len(case['lengths']) == 1
    bs = case.get('block_size', 64)
    with torch.inference_mode():
        qg, kg, vg = [x.cuda().unsqueeze(0) for x in (q, k, v)]
        kc = ENGINE.fla.mean_pooling(kg, bs, None)
        vc = ENGINE.fla.mean_pooling(vg, bs, None)
        _, lse = ENGINE.fla.parallel_nsa_compression_fwd(qg, kc, vc, bs, q.shape[-1] ** -0.5, None)
        selected = ENGINE.fla.parallel_nsa_topk(qg, kc, lse, case['selected_blocks'], bs, q.shape[-1] ** -0.5)
        idx = selected.squeeze(0).permute(1, 0, 2).contiguous().cpu()
    # The selector pads with BS-independent -1 sentinel; check its structural contract.
    valid = idx >= 0
    assert (idx[valid] < (sum(case['lengths']) + bs - 1) // bs).all()
    for row in rows:
        for head in range(idx.shape[0]):
            blocks = idx[head, row][valid[head, row]].tolist()
            assert len(blocks) == len(set(blocks)), (row, blocks)
            assert all(b <= row // bs for b in blocks), (row, blocks)
    h = hashlib.sha256()
    for t in (q, k, v, idx):
        h.update(t.view(torch.uint8).numpy().tobytes())
    return q, k, v, idx, rows, h.hexdigest()


def main():
    global ENGINE
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['validate', 'benchmark'])
    p.add_argument('--suite', choices=['all', 'edges', 'matrix', 'model-shapes'], default='all')
    p.add_argument('--names', nargs='+')
    p.add_argument('--providers', nargs='+', default=['fla', 'late_v_w4'])
    p.add_argument('--output', required=True)
    p.add_argument('--validation')
    p.add_argument('--rounds', type=int, default=7)
    a = p.parse_args()
    torch.set_num_threads(4)
    ENGINE = Experiments()
    harness.inputs = study_inputs
    work = [c for c in cases(a.suite) if not a.names or c['name'] in a.names]
    assert work
    if a.stage == 'validate':
        r = harness.validate(ENGINE, work, a.providers, a.output)
        assert all(c['status'] == 'passed' for c in r['cases']), 'See report for failures'
    else:
        assert a.validation
        gate = json.loads((harness.RESULTS / a.validation).read_text())
        assert gate['sources'] == ENGINE.sources
        valid = {(c['case']['name'], c['provider']): c['status'] for c in gate['cases']}
        assert all(valid.get((c['name'], arm)) == 'passed' for c in work for arm in a.providers)
        harness.benchmark(ENGINE, work, a.providers, gate, a.output, a.rounds)


if __name__ == '__main__':
    main()
