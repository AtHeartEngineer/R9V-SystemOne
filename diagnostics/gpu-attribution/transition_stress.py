#!/usr/bin/env python3
"""Blackbox-side, bounded transition stress for an already admitted R9V server."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
import traceback

IMAGE = 'sha256:977c0de0ee05c415a8f8fca7d38e78e5a9c8fbb083b1dad8e0a87f927ea370a0'
FILLER = 'The river flows past trees, hills, fields and quiet villages. '
MAX_RESPONSE = 2 * 1024 * 1024


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def guard(campaign, now=None):
    for name in ('STOP.json', 'gate-STOP.json', 'campaign-end.json', 'external-stop.json'):
        if (campaign / name).exists():
            raise RuntimeError('Campaign stopped: ' + name)
    status = json.loads((campaign / 'status.json').read_text())
    health = json.loads((campaign / 'capture-health.json').read_text())
    identity = json.loads((campaign / 'identity-admitted.json').read_text())
    config = json.loads((campaign / 'controller-config.json').read_text())
    # Sample after reading: an atomic publisher may advance timestamps during IO.
    now = time.time() if now is None else now
    if not (status['phase'] == 'running' and not status['fault'] and not health['fault']
            and 0 <= now - status['time'] < 10
            and 0 <= now - health['host_at'] < 10
            and 0 <= now - health['trace_at'] < 45):
        raise RuntimeError('Controller or independent telemetry is stale/unhealthy: ' + json.dumps({'now':now, 'phase':status['phase'], 'fault':status['fault'], 'status_age':now-status['time'], 'host_age':now-health['host_at'], 'trace_age':now-health['trace_at']}))
    pids = {tuple(w['namespace_pids']) for w in identity['workers']}
    if not (0 <= now - identity['time'] < 5 and len(pids) == 2
            and identity['boot_id'] == config['required_boot_id']):
        raise RuntimeError('Serving identity coverage lost: ' + json.dumps({'now':now, 'identity_age':now-identity['time'], 'worker_count':len(pids), 'boot':identity['boot_id']}))


def child(job):
    """One disposable HTTP process; parent enforces total wall-clock deadline."""
    req = urllib.request.Request(job['url'] + job['endpoint'],
                                 json.dumps(job['body']).encode(),
                                 {'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as response:
        if not job['body'].get('stream'):
            raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise ValueError('Response budget exceeded')
            data = json.loads(raw)
            if job['endpoint'] == '/tokenize':
                data = {'count': data['count']}
            return {'response': data, 'seconds': time.monotonic() - start}
        content, chunks, size, done, finish, usage = '', 0, 0, False, None, None
        for raw in response:
            size += len(raw)
            if size > MAX_RESPONSE or len(raw) > 262144:
                raise ValueError('SSE response budget exceeded')
            if not raw.startswith(b'data: '):
                continue
            if raw.strip() == b'data: [DONE]':
                done = True
                break
            row = json.loads(raw[6:])
            usage = row.get('usage') or usage
            for choice in row.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or delta.get('reasoning') or ''
                if text:
                    chunks += 1
                    content += text
                finish = choice.get('finish_reason') or finish
            if job.get('cancel_chunks') and chunks >= job['cancel_chunks']:
                return {'cancelled': True, 'content_chunks': chunks,
                        'seconds': time.monotonic() - start}
        if job.get('cancel_chunks'):
            raise ValueError('Stream ended before planned cancellation')
        if not done or finish not in ('stop', 'length', 'tool_calls'):
            raise ValueError('Truncated/incomplete SSE stream')
        return {'cancelled': False, 'content_chunks': chunks, 'finish': finish,
                'usage': usage, 'text_prefix': content[:100],
                'output_sha256': hashlib.sha256(content.encode()).hexdigest(),
                'seconds': time.monotonic() - start}


def execute(jobs, timeout, check=lambda: None):
    """Launch together to test queued requests; terminate every child on error."""
    children = []
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix='r9v-transition-') as tmp:
        try:
            for i, job in enumerate(jobs):
                path = Path(tmp) / f'{i}.json'
                path.write_text(json.dumps(job))
                proc = subprocess.Popen([sys.executable, __file__, '--child', str(path)],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                children.append(proc)
            while any(p.poll() is None for p in children):
                check()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Request group exceeded total deadline')
                time.sleep(.1)
            results = []
            for proc in children:
                out, err = proc.communicate(timeout=1)
                if proc.returncode:
                    raise RuntimeError('HTTP child failed: ' + err.decode()[-2000:])
                results.append(json.loads(out))
            return results
        finally:
            for proc in children:
                if proc.poll() is None:
                    proc.terminate()
            for proc in children:
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
                for pipe in (proc.stdout, proc.stderr):
                    pipe.close()


def prompt(seed, cycle, n):
    nonce = hashlib.sha256(f'{seed}:{cycle}:{n}'.encode()).hexdigest()
    return (f'Run {nonce}. Remember R9V-731. ' + FILLER * n
            + ' Repeat the code from the beginning. Reply only with that code.')


def body(model, text, tokens):
    return {'model': model, 'messages': [{'role': 'user', 'content': text}],
            'max_tokens': tokens, 'temperature': 0,
            'chat_template_kwargs': {'enable_thinking': False},
            'stream': True, 'stream_options': {'include_usage': True}}


def fitted_prompt(model, seed, cycle, target, call, prompt_builder=None):
    low, high, best = 0, max(1, target // 8), None
    while low <= high:
        n = (low + high) // 2
        text = (prompt_builder or prompt)(seed, cycle, n)
        count = call('/tokenize', {'model': model,
                     'messages': [{'role': 'user', 'content': text}],
                     'chat_template_kwargs': {'enable_thinking': False}})['response']['count']
        if count <= target:
            best = (text, count, n)
            low = n + 1
        else:
            high = n - 1
    if best is None or not target - 96 <= best[1] <= target:
        raise ValueError('Token calibration missed target envelope')
    return best


def scenario():
    return {'duration_seconds': 3600, 'minimum_cycles': 3,
            'round': ['short prompt / 1024 decode tokens', '32768-token prefill / 3 decode',
                      '130816-token prefill / 1 decode', 'short / 3 decode',
                      'cancel after 3 content-bearing SSE chunks; resume arithmetic',
                      'two simultaneous HTTP requests (one cancelled, one completed)',
                      'every third round: 30 seconds idle, then resume arithmetic'],
            'graphics': 'alternate one owned 512x512 30fps X11 control on/off each round',
            'transport_failure': 'freeze logs, request existing controller stop; no retry',
            'semantic_failure': 'record separately and continue',
            'power_actions': 'none', 'image': IMAGE}


def run(args):
    campaign, output = args.campaign.resolve(), args.output.resolve()
    guard(campaign)
    # This runner lives on blackbox and uses the existing independent controller.
    config = json.loads((campaign / 'controller-config.json').read_text())
    if len(config['containers']) != 1:
        raise ValueError('Exactly one owned model container required')
    container = config['containers'][0]
    remote = str(Path(config['remote_output']).parent)
    def ssh(*argv):
        import shlex
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o',
            'ConnectTimeout=5', 'workstation', shlex.join(argv)], text=True, timeout=12)
    paused = json.loads(ssh('cat', remote + '/BOUNDARY-PAUSED.json'))
    if paused.get('container') != container or paused.get('failed') != 0:
        raise RuntimeError('Harness must be paused cleanly between requests')
    info = json.loads(ssh('/home/dylan/bin/docker', 'inspect', container))[0]
    if info['Image'] != IMAGE or not info['State']['Running']:
        raise RuntimeError('Wrong or stopped model image')
    required = ['R9V_PLE_HOST_FENCE=1', 'R9V_STAGE_DIAGNOSTICS=0',
                'R9V_EVENT_BOUNDARIES=0', 'NCCL_P2P_DISABLE=0']
    if not all(x in info['Config']['Env'] for x in required):
        raise RuntimeError('Unexpected model instrumentation or transport')
    if any(m['Destination'].startswith('/opt/') for m in info['Mounts']):
        raise RuntimeError('Unexpected model source overlay')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'plan.json', dict(scenario(), seed=args.seed,
         requested_seconds=args.seconds, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    save(output / 'admission.json', {'time': time.time(), 'image': info['Image'],
         'container': container, 'paused': paused, 'boot': config['required_boot_id']})
    started, deadline = time.monotonic(), time.monotonic() + args.seconds
    unit_base = 'r9v-transition-graphics-' + uuid.uuid4().hex[:10]
    unit = unit_base
    gfx, cycles, groups, cancelled, semantics = False, 0, 0, 0, []
    def check():
        guard(campaign)
        if time.monotonic() >= deadline:
            raise TimeoutError('Whole test duration exhausted during a request')
    def event(record):
        with (output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), **record)) + '\n')
    def request(label, bodies, cancel=None, timeout=300, endpoint='/v1/chat/completions'):
        nonlocal groups, cancelled
        check()
        jobs = [dict(url=args.url.rstrip('/'), endpoint=endpoint, body=b,
                     cancel_chunks=(cancel or {}).get(i, 0)) for i, b in enumerate(bodies)]
        event({'event': 'start', 'label': label, 'cycle': cycles,
               'requests': [{'sha256': hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest(),
                             'cancel_chunks': j['cancel_chunks']} for j in jobs]})
        results = execute(jobs, min(timeout, deadline-time.monotonic()), check)
        groups += 1
        cancelled += sum(bool(r.get('cancelled')) for r in results)
        event({'event': 'complete', 'label': label, 'cycle': cycles, 'results': results})
        return results
    def arithmetic():
        r = request('resume', [body(args.model, 'Compute 17 + 25. Reply only with the integer.', 16)])[0]
        if r['text_prefix'].strip() != '42':
            semantics.append({'cycle': cycles, 'check': 'arithmetic', 'observed': r['text_prefix']})
    failure = None
    try:
        while deadline - time.monotonic() > 600 and cycles < 64:
            check()
            if cycles % 2 == 0:
                unit = unit_base + '-' + str(cycles)
                gfx = True  # Also clean up an ambiguously accepted SSH start.
                ssh('systemd-run', '--user', '--collect', '--unit=' + unit,
                    '--property=RuntimeMaxSec=75min', '--property=TimeoutStopSec=5',
                    '/usr/bin/python3', remote + '/x11_copy_workload.py')
                gfx = True
            elif gfx:
                ssh('systemctl', '--user', 'stop', unit)
                gfx = False
            event({'event': 'graphics', 'active': gfx, 'unit': unit})
            long_decode = body(args.model, f'Run {args.seed}-{cycles}. Write 200 numbered facts about rivers; do not conclude early.', 1024)
            request('long-decode', [long_decode])
            for target, tokens in [(32768, 3), (130816, 1)]:
                text, count, repeats = fitted_prompt(args.model, args.seed, cycles, target,
                    lambda endpoint, b: request('token-calibration', [b], endpoint=endpoint, timeout=30)[0])
                event({'event': 'shape', 'target': target, 'actual': count, 'repeats': repeats})
                r = request('prefill-' + str(target), [body(args.model, text, tokens)])[0]
                actual = (r.get('usage') or {}).get('prompt_tokens', 0)
                if not target - 96 <= actual <= target + 128:
                    raise RuntimeError('Server usage did not confirm prefill envelope')
            request('short-decode', [body(args.model, 'Say hello.', 3)])
            request('cancel-active', [long_decode], {0: 3})
            arithmetic()
            request('queued-disconnect', [long_decode, body(args.model, 'List five trees.', 64)], {0: 3})
            if gfx:
                state = ssh('systemctl', '--user', 'is-active', unit).strip()
                if state != 'active':
                    raise RuntimeError('Owned graphics workload stopped unexpectedly')
                log = ssh('journalctl', '--user', '-u', unit, '-n', '200', '--output=cat', '--no-pager')
                (output / f'graphics-{cycles}.log').write_text(log)
                if '"event": "progress"' not in log:
                    raise RuntimeError('Graphics progress not observed')
            if cycles % 3 == 2:
                until = time.monotonic() + 30
                event({'event': 'idle-start', 'cycle': cycles})
                while time.monotonic() < until:
                    check()
                    time.sleep(.25)
                arithmetic()
            cycles += 1
        # Keep collecting through the requested duration without starting an unbounded final round.
        while time.monotonic() < deadline:
            guard(campaign)
            time.sleep(.25)
        if cycles < 3:
            raise RuntimeError('Insufficient completed transition cycles')
    except BaseException as error:
        failure = repr(error)
        event({'event': 'failure', 'error': failure, 'traceback': traceback.format_exc()})
    finally:
        # Let the existing controller freeze first, stop the model, and retain aftermath.
        marker = campaign / 'external-stop.json'
        if not marker.exists():
            save(marker, {'time': time.time(), 'reason': 'Transition stress ' + (failure or 'completed normally')})
        cleanup_error = None
        try:
            if gfx:
                ssh('systemctl', '--user', 'stop', unit)
        except Exception as error:
            cleanup_error = repr(error)
        save(output / 'result.json', {'stability_passed': failure is None and cleanup_error is None,
             'failure': failure, 'graphics_cleanup_error': cleanup_error, 'cycles': cycles,
             'request_groups': groups, 'planned_disconnections': cancelled,
             'semantic_errors': semantics, 'seconds': time.monotonic()-started,
             'aftermath_verified': False, 'note': 'Controller campaign-end and 90-second aftermath must be reviewed separately.'})
    return 1 if failure or cleanup_error else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--child', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--campaign', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--url', default='http://192.168.1.231:8004')
    p.add_argument('--model', default='qwen3.8-flash-next')
    p.add_argument('--seconds', type=int, default=3600)
    p.add_argument('--seed', type=int, default=731)
    args = p.parse_args()
    if args.child:
        print(json.dumps(child(json.loads(args.child.read_text()))))
        return 0
    if args.plan:
        print(json.dumps(scenario(), indent=2))
        return 0
    if not args.campaign or not args.output or not 900 <= args.seconds <= 7200:
        p.error('--campaign and fresh --output required; --seconds must be 900..7200')
    def stop(*_):
        raise KeyboardInterrupt('Owned test interrupted')
    signal.signal(signal.SIGTERM, stop)
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
