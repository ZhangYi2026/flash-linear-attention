import sys,json
from pathlib import Path
import xml.etree.ElementTree as ET
root=ET.parse(sys.argv[1]).getroot()
cases=list(root.iter('testcase'))
path=Path(sys.argv[1])
expected=json.loads(path.with_suffix('.nodes.json').read_text())
assert len(cases)==len(expected)>0, (len(cases),len(expected))
assert json.loads(path.with_suffix('.exit.json').read_text())['exitstatus']==0, 'Pytest did not complete successfully'
failed=[c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]
assert not failed, failed
print('Gate passed:',sys.argv[1],flush=True)
