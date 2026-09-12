import json,hashlib,struct
from pathlib import Path
root=Path(__file__).resolve().parent
workers=[]
for device in range(2):
 ready=json.loads((root/('smoke-ready-'+str(device)+'.json')).read_text())
 objects=[]
 for p in sorted((root/('debug-'+str(ready['pid']))).glob('object-*.elf')):
  data=p.read_bytes()
  objects.append(dict(file=str(p.relative_to(root)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),elf=data[:4]==b'\x7fELF',machine=struct.unpack_from('<H',data,18)[0] if len(data)>20 else None))
 workers.append(dict(device=device,pid=ready['pid'],objects=objects))
print(json.dumps(dict(passed=all(w['objects'] and all(o['elf'] and o['machine']==224 for o in w['objects']) for w in workers),workers=workers)))
