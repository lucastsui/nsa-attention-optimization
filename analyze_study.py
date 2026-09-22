"""Summarize recorded sessions without combining their clocks or hiding variation."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def read(name):
    return json.loads((RESULTS / name).read_text())


def main():
    validation = read('study_combined_validation.json')
    assert validation['status'] == 'completed'
    assert all(c['status'] == 'passed' for c in validation['cases'])
    out = dict(validation_comparisons=len(validation['cases']),
               max_relative_l2=max(c['relative_l2'] for c in validation['cases']),
               max_abs_error=max(c['max_abs_error'] for c in validation['cases']),
               sessions=[], scope='BF16 selected forward; G16 Hkv4 D128 block64 top16; synthetic QKV')
    for name in ('study_session1.json', 'study_session2.json'):
        data = read(name)
        assert data['status'] == 'completed' and data['sources'] == validation['sources']
        lookup = {(c['case']['name'], c['provider']): c for c in data['cases']}
        records = []
        for case in sorted({c['case']['name'] for c in data['cases']}):
            cand = lookup[case, 'late_v_w4']
            for control in ('fla', 'fla_tuned'):
                base = lookup[case, control]
                assert base['input_sha256'] == cand['input_sha256']
                assert base['status'] == cand['status'] == 'measured'
                assert len(base['wall_ms']) == len(cand['wall_ms']) == 21
                rounds = [st.median(base['wall_ms'][i:i+3])/st.median(cand['wall_ms'][i:i+3]) for i in range(0,21,3)]
                records.append(dict(case=case, control=control, baseline_ms=base['median_wall_ms'],
                    candidate_ms=cand['median_wall_ms'], speedup=base['median_wall_ms']/cand['median_wall_ms'],
                    round_speedups=rounds, min_round_speedup=min(rounds), max_round_speedup=max(rounds),
                    faster_rounds=sum(x>1 for x in rounds), baseline_resources=base['compiled_resources'],
                    candidate_resources=cand['compiled_resources']))
        out['sessions'].append(dict(file=name, timestamp_utc=data['timestamp_utc'], comparisons=records))
    out['gain_vs_tuned_percent_range'] = [100*(f(r['speedup'] for s in out['sessions'] for r in s['comparisons'] if r['control']=='fla_tuned')-1) for f in (min,max)]
    (RESULTS/'study_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='sessions'},indent=2))


if __name__ == '__main__':
    main()
