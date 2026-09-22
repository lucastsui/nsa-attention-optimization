"""Run all sanitizer modes and instrumentation controls, retaining tool exit codes."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    a = p.parse_args()
    assert a.tag and all(c.isalnum() or c in '_-' for c in a.tag)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'{a.tag}_runs.json'
    assert not out.exists(), 'Choose a new tag to preserve prior evidence'
    record = dict(started_utc=datetime.now(timezone.utc).isoformat(), status='running', runs=[])
    for mode in ('memcheck', 'racecheck', 'synccheck', 'initcheck'):
        cmd = ['bash', str(HERE / 'sanitize.sh'), mode, a.tag]
        console = RESULTS / f'{a.tag}_{mode}_console.log'
        print('STARTING ' + mode, flush=True)
        entry = dict(tool=mode, command=cmd, started_utc=datetime.now(timezone.utc).isoformat(),
                     kernel_filter=None if mode == 'initcheck' else 'kns=parallel_nsa_fwd_kernel')
        with console.open('w') as stream:
            run = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        entry.update(exit_code=run.returncode, finished_utc=datetime.now(timezone.utc).isoformat(),
                     console_log=console.name, tool_log=f'{a.tag}_{mode}.log',
                     validation=f'{a.tag}_{mode}_validation.json', saved_session=f'{a.tag}_{mode}.san')
        record['runs'].append(entry)
        out.write_text(json.dumps(record, indent=2) + '\n')
        print(f'FINISHED {mode}: exit={run.returncode}', flush=True)
        if run.returncode:
            record['status'] = 'needs_investigation'
            out.write_text(json.dumps(record, indent=2) + '\n')
            raise SystemExit(run.returncode)
    record.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    out.write_text(json.dumps(record, indent=2) + '\n')
    controls = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), runs=[])
    control_path = RESULTS / f'{a.tag}_controls.json'
    sanitizer = os.environ.get('COMPUTE_SANITIZER', '/usr/local/cuda-12.8/bin/compute-sanitizer')
    for mode in ('valid', 'invalid_read'):
        log = RESULTS / f'{a.tag}_control_{mode}.log'
        console = RESULTS / f'{a.tag}_control_{mode}_console.log'
        cmd = [sanitizer, '--tool', 'memcheck', '--error-exitcode', '97', '--target-processes', 'all',
               '--kernel-name', 'kns=parallel_nsa_fwd_kernel', '--log-file', str(log),
               'bash', str(HERE / 'run.sh'), 'sanitizer_probe.py']
        if mode == 'invalid_read':
            cmd.append('--bad')
        with console.open('w') as stream:
            run = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        text = log.read_text()
        passed = (run.returncode == 0 and 'ERROR SUMMARY: 0 errors' in text if mode == 'valid'
                  else run.returncode != 0 and 'Invalid __global__ read of size 4 bytes' in text)
        controls['runs'].append(dict(control=mode, command=cmd, exit_code=run.returncode,
                                    expected_behavior_verified=passed, tool_log=log.name, console_log=console.name))
        control_path.write_text(json.dumps(controls, indent=2) + '\n')
        assert passed, 'Instrumentation control did not behave as expected'
    controls['status'] = 'passed'
    control_path.write_text(json.dumps(controls, indent=2) + '\n')
    subprocess.run([sys.executable, str(HERE / 'verify_sanitizers.py'), '--tag', a.tag,
                    '--controls', control_path.name, '--output', f'{a.tag}_summary.json'], check=True)


if __name__ == '__main__':
    main()
