import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from host_capture import identities, copy_dump
from trace_check import TraceCheck
spec = importlib.util.spec_from_file_location('capture_hook', Path(__file__).with_name('sitecustomize.py'))
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class CaptureTests(unittest.TestCase):
    def test_protected_desktop_identity_and_pid_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            p = proc / '123'
            (p / 'fd').mkdir(parents=True)
            (p / 'fdinfo').mkdir()
            (p / 'fd/7').symlink_to('/dev/dri/card1')
            (p / 'status').write_text('Name:\tkwin_wayland\nUid:\t1000 1000 1000 1000\nNSpid:\t123\n')
            (p / 'stat').write_text('123 (name with ) spaces) ' + ' '.join(['S'] + ['0'] * 18 + ['98765']))
            (p / 'fdinfo/7').write_text('drm-pdev:\t0000:03:00.0\npasid:\t23\ndrm-client-id:\t19\n')
            result = identities(proc)
            self.assertFalse(result['errors'])
            self.assertEqual(result['rows'][0]['pasid'], 23)
            self.assertEqual(result['rows'][0]['start_ticks'], 98765)

    def test_dump_exact_copy_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data'
            source.write_bytes(b'fixture-device-dump')
            result = copy_dump(source, root / 'saved.dump', 100)
            self.assertEqual(result['bytes'], 19)
            self.assertEqual((root / 'saved.dump').read_bytes(), source.read_bytes())
            with self.assertRaises(ValueError):
                copy_dump(source, root / 'limited.dump', 8)
            self.assertFalse((root / 'limited.dump').exists())

    def test_only_decode_stall_is_signalled(self):
        row = dict(pid=10, stage='execute_model_enter', steps=50, last_batch_tokens=3, monotonic_ns=1)
        self.assertTrue(hook.stalled(row, 10, 2_000_000_001))
        for updates in ({'last_batch_tokens':1024}, {'stage':'execute_model_return'}, {'pid':11}, {'steps':1}):
            self.assertFalse(hook.stalled(dict(row, **updates), 10, 2_000_000_001))
        self.assertFalse(hook.stalled(row, 10, 100))

    def test_enqueue_exit_is_not_completion_and_gaps_fail(self):
        check = TraceCheck()
        for row in [dict(pid=1, seq=1, type='ready'),
                    dict(pid=1, seq=2, type='enqueue', dispatch=9, name='kernel'),
                    dict(pid=1, seq=4, type='symbol')]:
            check.feed('timestamp R9V_GPU ' + json.dumps(row))
        self.assertEqual(len(check.result()['unfinished']), 1)
        self.assertTrue(check.gaps)
        with self.assertRaises(ValueError):
            check.validate_smoke()


if __name__ == '__main__':
    unittest.main()
