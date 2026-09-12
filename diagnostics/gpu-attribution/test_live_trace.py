import json
import unittest
from live_trace import LiveTrace

class LiveTraceTests(unittest.TestCase):
    def feed(self, trace, pid, seq, **fields):
        trace.feed('timestamp R9V_GPU '+json.dumps(dict(pid=pid,seq=seq,**fields)))

    def test_interleaved_workers_keep_distinct_unfinished_dispatches(self):
        trace=LiveTrace()
        for pid in (10,11):
            self.feed(trace,pid,1,type='object_saved',code_object=1,path='object.elf')
            self.feed(trace,pid,2,type='enqueue',code_object=1,dispatch=8,name='kernel')
        self.feed(trace,11,3,type='complete',dispatch=8)
        self.assertEqual([(r['pid'],r['dispatch']) for r in trace.snapshot()['unfinished']],[(10,8)])
        self.assertEqual(trace.completions,1)
        self.assertFalse(trace.errors)

    def test_other_worker_object_cannot_admit_dispatch(self):
        trace=LiveTrace()
        self.feed(trace,10,1,type='object_saved',code_object=1)
        self.feed(trace,11,1,type='enqueue',code_object=1,dispatch=8,name='kernel')
        self.assertIn('enqueue without captured code object',trace.errors)

    def test_explicit_error_advances_sequence_without_fake_gap(self):
        trace=LiveTrace()
        self.feed(trace,10,1,error='code object copy failed')
        self.feed(trace,10,2,type='finalize')
        self.assertEqual(trace.errors,['code object copy failed'])

if __name__=='__main__':unittest.main()
