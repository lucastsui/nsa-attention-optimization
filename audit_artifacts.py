"""Audit packaged files and recorded evidence without running GPU code."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / 'ARTIFACT_MANIFEST.json').read_text())
    files = manifest['files']
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), f'Path outside repository: {name}'
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f'Artifact hash mismatch: {name}'
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=name)
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
        elif path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if '://' in target or target.startswith(('#', '/')):
                    continue
                clean = target.split('#', 1)[0].split(' "', 1)[0]
                assert not clean or (path.parent / clean).exists(), (name, target)

    if (ROOT / '.git').exists():
        tracked = set(subprocess.check_output(
            ['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines())
        assert tracked == set(files) | {'ARTIFACT_MANIFEST.json'}, 'Manifest coverage differs from tracked files'

    # Write the regenerated summary outside the tree to preserve recorded evidence.
    with tempfile.TemporaryDirectory(prefix='nsa-evidence-') as temp:
        subprocess.run([
            sys.executable, str(ROOT / 'verify_sanitizers.py'),
            '--output', str(Path(temp) / 'sanitizer_summary.json'),
        ], check=True)
    print(f'PASS: {len(files)} file hashes, source syntax, local links, and recorded sanitizer evidence.')
    print('This audit does not execute GPU kernels or reproduce performance measurements.')


if __name__ == '__main__':
    main()
