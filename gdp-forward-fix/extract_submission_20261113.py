from pathlib import Path
import tarfile,json,hashlib
root=Path(__file__).resolve().parent
for row in json.loads((root/'submission-source-manifest.json').read_text()):
 archive=root/(row['name']+'.tar')
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['archive_sha256']
 dest=Path('/home/zhang_e/.cache/fla-rewy-pr-20261108')/row['name']
 assert not dest.exists()
 dest.mkdir()
 with tarfile.open(archive) as t:t.extractall(dest,filter='data')
 row['files']={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['fla','tests'] for p in (dest/folder).rglob('*.py')}
 (root/(row['name']+'-verified.json')).write_text(json.dumps(row,indent=2)+'\n')
 print(row['name'],row['commit'],len(row['files']))
