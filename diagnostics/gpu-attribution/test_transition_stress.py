import http.server
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import transition_stress as stress


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *_):
        pass

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200)
        self.end_headers()
        try:
            if self.path == '/stall':
                time.sleep(2)
                return
            if self.path == '/tokenize':
                # Large raw tokenizer result must not fill the child stdout pipe.
                self.wfile.write(json.dumps({'count': 130816, 'tokens': [1]*130816}).encode())
                return
            for i in range(5):
                row = {'choices': [{'delta': {'content': str(i)}, 'finish_reason': None}]}
                self.wfile.write(('data: ' + json.dumps(row) + '\n\n').encode())
                self.wfile.flush()
                time.sleep(.03)
            if self.path != '/truncated':
                self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        except (BrokenPipeError, ConnectionResetError):
            pass


class TransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def job(self, endpoint='/ok', cancel=0, stream=True):
        return dict(url=self.url, endpoint=endpoint, body={'stream': stream}, cancel_chunks=cancel)

    def test_cancellation_and_simultaneous_completion_are_distinct(self):
        cancelled, complete = stress.execute([self.job(cancel=3), self.job()], 4)
        self.assertTrue(cancelled['cancelled'])
        self.assertEqual(cancelled['content_chunks'], 3)
        self.assertFalse(complete['cancelled'])
        self.assertEqual(complete['text_prefix'], '01234')

    def test_truncated_stream_is_not_a_successful_cancellation(self):
        with self.assertRaisesRegex(RuntimeError, 'Truncated/incomplete'):
            stress.execute([self.job('/truncated')], 4)

    def test_early_end_does_not_count_as_planned_disconnect(self):
        with self.assertRaisesRegex(RuntimeError, 'before planned cancellation'):
            stress.execute([self.job(cancel=10)], 4)

    def test_stalled_http_has_a_total_deadline(self):
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            stress.execute([self.job('/stall')], .3)
        self.assertLess(time.monotonic()-start, 1.5)

    def test_coverage_loss_interrupts_an_active_request(self):
        def missing():
            raise RuntimeError('identity expired')
        with self.assertRaisesRegex(RuntimeError, 'identity expired'):
            stress.execute([self.job('/stall')], 4, missing)

    def test_token_array_cannot_deadlock_child_pipe(self):
        result = stress.execute([self.job('/tokenize', stream=False)], 4)[0]
        self.assertEqual(result['response'], {'count': 130816})

    def test_real_count_controls_prompt_size(self):
        def count(endpoint, request):
            self.assertEqual(endpoint, '/tokenize')
            return {'response': {'count': len(request['messages'][0]['content'])//4}}
        text, tokens, repeats = stress.fitted_prompt('model', 731, 0, 8192, count)
        self.assertLessEqual(tokens, 8192)
        self.assertGreaterEqual(tokens, 8096)
        self.assertEqual(text, stress.prompt(731, 0, repeats))
        self.assertNotEqual(text, stress.prompt(731, 1, repeats))

    def test_atomic_writer_update_during_read_is_fresh(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            clock = [100.0]
            stress.save(p/'status.json', dict(phase='running', fault=None, time=100, host_at=100, trace_at=100))
            stress.save(p/'capture-health.json', dict(host_at=100, trace_at=100, fault=None))
            stress.save(p/'identity-admitted.json', dict(time=100, boot_id='boot', workers=[{'namespace_pids':[1,10]}, {'namespace_pids':[2,20]}]))
            stress.save(p/'controller-config.json', {'required_boot_id':'boot'})
            original = Path.read_text
            def read(path, *args, **kwargs):
                if path.name == 'status.json':
                    clock[0] = 100.001
                    stress.save(path, dict(phase='running', fault=None, time=clock[0], host_at=clock[0], trace_at=100))
                return original(path, *args, **kwargs)
            with patch.object(stress.time, 'time', side_effect=lambda: clock[0]), patch.object(Path, 'read_text', read):
                stress.guard(p)

    def test_primary_receipt_not_periodic_summary_controls_freshness(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            stress.save(p/'status.json', dict(phase='running', fault=None, time=105, host_at=100, trace_at=80))
            stress.save(p/'identity-admitted.json', dict(time=110, boot_id='boot', workers=[{'namespace_pids':[1,10]}, {'namespace_pids':[2,20]}]))
            stress.save(p/'controller-config.json', {'required_boot_id':'boot'})
            stress.save(p/'capture-health.json', dict(host_at=110, trace_at=107, fault=None))
            stress.guard(p, 110.02)
            for health in [dict(host_at=100, trace_at=107, fault=None), dict(host_at=110, trace_at=65, fault=None), dict(host_at=110, trace_at=107, fault={'timeout':True})]:
                stress.save(p/'capture-health.json', health)
                with self.assertRaises(RuntimeError):
                    stress.guard(p, 110.02)

    def test_publication_preserves_old_snapshot_until_new_is_complete(self):
        from unittest.mock import patch
        import atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'status.json'
            atomic_json.save_json(path, {'version':1})
            original = atomic_json.json.dump
            def during_write(value, stream, **kwargs):
                self.assertEqual(json.loads(path.read_text()), {'version':1})
                original(value, stream, **kwargs)
                self.assertEqual(json.loads(path.read_text()), {'version':1})
            with patch.object(atomic_json.json, 'dump', during_write):
                atomic_json.save_json(path, {'version':2})
            self.assertEqual(json.loads(path.read_text()), {'version':2})
            with patch.object(atomic_json.json, 'dump', side_effect=OSError('write failed')):
                with self.assertRaises(OSError):
                    atomic_json.save_json(path, {'version':3})
            self.assertEqual(json.loads(path.read_text()), {'version':2})
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_stop_reboot_stale_and_missing_worker_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            status = dict(phase='running', fault=None, time=100, host_at=100, trace_at=100)
            identity = dict(time=100, boot_id='boot', workers=[{'namespace_pids':[1,10]}, {'namespace_pids':[2,20]}])
            def write():
                stress.save(p/'status.json', status)
                stress.save(p/'capture-health.json', dict(host_at=100, trace_at=100, fault=None))
                stress.save(p/'identity-admitted.json', identity)
                stress.save(p/'controller-config.json', {'required_boot_id':'boot'})
            write()
            stress.guard(p, 101)
            with self.assertRaises(RuntimeError):
                stress.guard(p, 111)
            identity['boot_id'] = 'new-boot'; write()
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                stress.guard(p, 101)
            identity.update(boot_id='boot', workers=[{'namespace_pids':[1,10]}]); write()
            with self.assertRaises(RuntimeError):
                stress.guard(p, 101)
            stress.save(p/'STOP.json', {'reason':'prior failure'})
            with self.assertRaisesRegex(RuntimeError, 'Campaign stopped'):
                stress.guard(p, 101)


if __name__ == '__main__':
    unittest.main()
