from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / 'gate_xml.py'), str(root / 'kda-fixture-fix-20261113.xml')], check=True)
repo = root.parent / 'fla_gdp_evidence_20261108'
target = repo / 'kda-cache-inputs'
assert not target.exists()
target.mkdir()
selected = [p for p in root.iterdir() if p.is_file() and (p.name.startswith(('kda-control-20261113-', 'kda-fixture-fix-20261113.')) or p.name in {'KDA_CACHE_ROOT_CAUSE.md', 'diagnose_kda_20261113.py', 'diagnose_kda_20261113.sh', 'kda-diagnosis-20261113.json', 'run_kda_control_20261113.py', 'run_kda_fixture_fix_20261113.sh', 'audit_collection.py', 'gate_xml.py'})]
manifest = []
for p in sorted(selected):
    dest = target / ('README.md' if p.name == 'KDA_CACHE_ROOT_CAUSE.md' else p.name)
    shutil.copy2(p, dest)
    manifest.append({'file': dest.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
(target / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
readme = repo / 'README.md'
readme.write_text(readme.read_text(encoding='utf-8') + '\n## Independent cache-test fixture investigation\n\n[KDA cache-test overflow and bounded input correction](kda-cache-inputs/README.md). This test-only correction is independent of the ReWY backend and shared-solver scheduling fix.\n', encoding='utf-8', newline='\n')
print(json.dumps({'files': len(manifest), 'bytes': sum(r['bytes'] for r in manifest)}, indent=2))
