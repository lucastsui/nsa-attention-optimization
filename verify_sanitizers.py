"""Verify tool exit/logs, numerical outputs, source hashes, and instrumentation controls."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
EXPECTED={'edge0_s29','edge1_s29','edge2_s29','edge3_s29','edge4_s29','edge5_s29',
          'edge6_s29','edge7_s29','n8192_random_s29','n8192_nsa_s29'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tag',default='sanitizer_final_20260922')
    p.add_argument('--controls',default='sanitizer_controls.json')
    p.add_argument('--output',default='sanitizer_summary.json')
    a=p.parse_args()
    runs=json.loads((RESULTS/f'{a.tag}_runs.json').read_text())
    controls=json.loads((RESULTS/a.controls).read_text())
    assert runs['status']=='completed' and controls['status']=='passed'
    assert {r['tool'] for r in runs['runs']}=={'memcheck','racecheck','synccheck','initcheck'}
    assert all(r['expected_behavior_verified'] for r in controls['runs'])
    assert {r['control'] for r in controls['runs']}=={'valid','invalid_read'}
    for r in controls['runs']:
        log=(RESULTS/r['tool_log']).read_text()
        if r['control']=='valid':
            assert r['exit_code']==0 and 'ERROR SUMMARY: 0 errors' in log
        else:
            assert r['exit_code']!=0 and 'Invalid __global__ read of size 4 bytes' in log
    original=json.loads((RESULTS/'study_combined_validation.json').read_text())
    sources=original['sources']
    expected_cases={c['case']['name']:c for c in original['cases'] if c['provider']=='late_v_w4'}
    assert hashlib.sha256((HERE/'candidates/late_v.py').read_bytes()).hexdigest()==sources['late_v_candidate']['sha256']
    summary=dict(verified_utc=datetime.now(timezone.utc).isoformat(),status='passed',
                 source=sources, tools=[], regenerated_selection_variants=[], instrumentation_control='passed; valid read clean, invalid read detected',
                 scope='Ten cases per tool, full launched grids. Memcheck/racecheck/synccheck filter for selected-forward kernels. Initcheck includes all kernels to track initialization through helper kernels and the selector.')
    for r in runs['runs']:
        log=(RESULTS/r['tool_log']).read_text()
        validation=json.loads((RESULTS/r['validation']).read_text())
        clean_marker=('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                      if r['tool']=='racecheck' else 'ERROR SUMMARY: 0 errors')
        assert r['exit_code']==0 and clean_marker in log, (r['tool'], log)
        assert not any(x in log for x in ('Failed to initialize','Device not supported','No attachable process','Warning:'))
        assert validation['status']=='completed' and validation['sources']==sources
        assert len(validation['cases'])==10
        assert {c['case']['name'] for c in validation['cases']}==EXPECTED
        assert all(c['status']=='passed' and c['provider']=='late_v_w4' for c in validation['cases'])
        for c in validation['cases']:
            original_case=expected_cases[c['case']['name']]
            assert c['case']==original_case['case']
            if c['input_sha256']!=original_case['input_sha256']:
                # Algorithm-generated indices are regenerated under instrumentation.
                # Each run's own indices are used by its independent FP64 reference.
                assert c['case']['pattern']=='nsa', 'Fixed-input cases must remain byte-identical'
                summary['regenerated_selection_variants'].append(dict(tool=r['tool'],case=c['case']['name'],
                    input_sha256=c['input_sha256'],original_input_sha256=original_case['input_sha256'],
                    note='Regenerated algorithm-selected workload; reference recomputed for the actual indices. This is not a timing-input substitution.'))
            if 'kernel_name' in c['compiled_resources']:
                assert 'parallel_nsa_fwd_kernel' in c['compiled_resources']['kernel_name']
        summary['tools'].append(dict(tool=r['tool'],exit_code=0,reported_errors=0,checked_cases=10,
                                    numerical_rechecks_passed=10,log=r['tool_log'],validation=r['validation']))
    summary['numerical_rechecks']=40
    (RESULTS/a.output).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='source'},indent=2))


if __name__=='__main__': main()
