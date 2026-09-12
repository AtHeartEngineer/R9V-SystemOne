import http.server,json,threading,time,unittest
import release_stress as stress
from test_transition_stress import Handler

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def job(self,endpoint='/ok',**kw):return dict(url=self.url,endpoint=endpoint,body={'stream':True},**kw)
    def test_eight_clients_and_distinct_drop_classes(self):
        jobs=[self.job() for _ in range(4)]+[self.job(cancel_chunks=1),self.job(cancel_chunks=3),self.job('/stall',disconnect_after=.05),self.job('/stall',disconnect_after=.1)]
        results=stress.execute(jobs,5)
        self.assertEqual(sum(not r.get('cancelled') for r in results),4)
        self.assertEqual([r.get('cancellation') for r in results[4:]],['content-confirmed-stream-drop']*2+['timed-client-drop']*2)
    def test_eight_requests_reach_server_together(self):
        barrier=threading.Barrier(8,timeout=3)
        class Together(Handler):
            def do_POST(self):
                barrier.wait();super().do_POST()
        self.server.RequestHandlerClass=Together
        self.assertEqual(len(stress.execute([self.job() for _ in range(8)],5)),8)
    def test_server_failure_is_not_counted_as_cancellation(self):
        with self.assertRaisesRegex(RuntimeError,'HTTP child'):
            stress.execute([self.job('/truncated')],3)
    def test_completed_request_cannot_count_as_timed_drop(self):
        with self.assertRaisesRegex(RuntimeError,'ended before planned drop'):
            stress.execute([self.job(disconnect_after=2)],3)
    def test_deadline_and_lost_coverage_stop_all_children(self):
        with self.assertRaises(TimeoutError):stress.execute([self.job('/stall') for _ in range(8)],.3)
        def lost():raise RuntimeError('coverage lost')
        with self.assertRaisesRegex(RuntimeError,'coverage lost'):stress.execute([self.job('/stall') for _ in range(8)],3,lost)
    def test_explicit_retrieval_oracle_is_used_during_calibration(self):
        import transition_stress as base
        def count(endpoint,body):
            text=body['messages'][0]['content']
            self.assertIn('The retrieval target is exactly R9V-731.',text)
            self.assertTrue(text.endswith('Do not return the run identifier.'))
            return {'response':{'count':len(text)//4}}
        text,tokens,n=base.fitted_prompt('model',7,0,8192,count,prompt_builder=stress.retrieval_prompt)
        self.assertEqual(text,stress.retrieval_prompt(7,0,n))
        self.assertTrue(8096<=tokens<=8192)

    def test_bounded_client_count(self):
        for count in [0,9]:
            with self.assertRaises(ValueError):stress.execute([self.job() for _ in range(count)],3)
if __name__=='__main__':unittest.main()
