import json
from pathlib import Path
import tempfile
import unittest
from serving_capture import read_workers

class ServingCaptureTests(unittest.TestCase):
    def test_dispatch_library_suffices_without_debugger(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stages=root/'stages';stages.mkdir();proc=root/'proc';proc.mkdir()
            for rank,pid in enumerate((10,11)):
                (proc/str(pid)).mkdir()
                (proc/str(pid)/'maps').write_text('/capture/libgpu_trace.so\n')
                (stages/('r9v-stage-'+str(pid)+'.json')).write_text(json.dumps(dict(pid=pid,rank=rank)))
            self.assertEqual({r['rank'] for r in read_workers(stages,proc)},{0,1})
            (proc/'11'/'maps').write_text('/opt/rocm/lib/librocm-debug-agent.so\n')
            self.assertEqual({r['rank'] for r in read_workers(stages,proc)},{0})
            (proc/'10'/'maps').write_text('/lib/libc.so\n')
            self.assertEqual(read_workers(stages,proc),[])

if __name__=='__main__':unittest.main()
