#!/usr/bin/env python3
"""Bounded eight-client release stress against the pinned shipping configuration."""
import argparse, hashlib, json, os, random, signal, subprocess, sys, tempfile, time, traceback
from pathlib import Path
import transition_stress as base


def retrieval_prompt(seed, cycle, n):
    nonce = hashlib.sha256(f'{seed}:{cycle}:{n}'.encode()).hexdigest()
    return (f'Run identifier (not the retrieval target): {nonce}. The retrieval target is exactly R9V-731. '
            + base.FILLER*n
            + ' Return only the exact value labeled retrieval target. Do not return the run identifier.')


def execute(jobs, timeout, check=lambda: None):
    if not 1 <= len(jobs) <= 8:
        raise ValueError('One to eight jobs required')
    processes, outcomes = [], {}
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='r9v-release-http-') as directory:
        try:
            for i, job in enumerate(jobs):
                path = Path(directory)/f'{i}.json'
                path.write_text(json.dumps(job))
                processes.append(subprocess.Popen([sys.executable, str(Path(base.__file__)), '--child', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE))
            while len(outcomes) < len(processes):
                check()
                elapsed = time.monotonic()-start
                if elapsed > timeout:
                    raise TimeoutError('Eight-client wave exceeded total deadline')
                for i, proc in enumerate(processes):
                    if i in outcomes:
                        continue
                    # Timed drops are separately classified: no claim that server work started.
                    if proc.poll() is None and jobs[i].get('disconnect_after') is not None and elapsed >= jobs[i]['disconnect_after']:
                        proc.terminate()
                        try:proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=2)
                        out, err = proc.communicate()
                        outcomes[i] = dict(cancelled=True, cancellation='timed-client-drop', seconds=elapsed, server_started_unverified=True)
                    elif proc.poll() is not None:
                        out, err = proc.communicate()
                        if proc.returncode:
                            raise RuntimeError(f'HTTP child {i} failed: '+err.decode()[-3000:])
                        result = json.loads(out)
                        if jobs[i].get('disconnect_after') is not None:
                            raise RuntimeError('Timed-drop request ended before planned drop')
                        if result.get('cancelled'):result['cancellation']='content-confirmed-stream-drop'
                        outcomes[i]=result
                time.sleep(.05)
            return [outcomes[i] for i in range(len(jobs))]
        finally:
            for proc in processes:
                if proc.poll() is None:proc.terminate()
            for proc in processes:
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=2)
                proc.communicate()


def run(args):
    campaign, output = args.campaign.resolve(), args.output.resolve()
    base.guard(campaign)
    config=json.loads((campaign/'controller-config.json').read_text())
    container=config['containers'][0];assert len(config['containers'])==1
    remote=str(Path(config['remote_output']).parent)
    import shlex
    def ssh(*argv):
        return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','ServerAliveInterval=3','-o','ServerAliveCountMax=2','workstation',shlex.join(argv)],text=True,timeout=15)
    info=json.loads(ssh('/home/dylan/bin/docker','inspect',container))[0]
    paused=json.loads(ssh('cat',remote+'/BOUNDARY-PAUSED.json'))
    assert paused['completed']==2 and paused['failed']==0 and paused['container']==container
    assert info['Image']==base.IMAGE and info['State']['Running']
    assert all(x in info['Config']['Env'] for x in ['R9V_PLE_HOST_FENCE=1','R9V_STAGE_DIAGNOSTICS=0','R9V_EVENT_BOUNDARIES=0','NCCL_P2P_DISABLE=0'])
    assert not any(m['Destination'].startswith('/opt/') for m in info['Mounts'])
    output.mkdir(parents=True,exist_ok=False)
    base.save(output/'admission.json',dict(time=time.time(),image=info['Image'],container=container,boot=config['required_boot_id'],paused=paused))
    base.save(output/'plan.json',dict(seconds=args.seconds,seed=args.seed,max_clients=8,maximum_wave_seconds=600,minimum_waves=8,final_idle_seconds=0,graphics='1920x1080 X11 copies at60fps, alternated each wave',runtime_image=base.IMAGE,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    started=time.monotonic();waves=groups=0;errors=[];failure=None;gfx=False
    unit='r9v-release-graphics-'+str(os.getpid())
    rng=random.Random(args.seed)
    def event(row):
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**row))+'\n')
    def check():
        base.guard(campaign)
        # Independent durable copy must keep up too; do not silently rely on RAM only.
        archive=json.loads((campaign/'archive-ack.json').read_text());now=time.time()
        if not 0<=now-archive['time']<90:raise RuntimeError('Secondary evidence archive stale')
        if time.monotonic()-started>args.seconds+650:raise TimeoutError('Whole-test overrun limit')
    def request(label,jobs,timeout=600):
        nonlocal groups
        check();event(dict(event='start',wave=waves,label=label,jobs=[dict(sha256=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest(),cancel_chunks=j.get('cancel_chunks',0),disconnect_after=j.get('disconnect_after')) for j in jobs]))
        results=execute(jobs,timeout,check);groups+=1
        event(dict(event='complete',wave=waves,label=label,results=results));return results
    def job(text,tokens=64,**kwargs):
        return dict(url=args.url,endpoint='/v1/chat/completions',body=base.body(args.model,text,tokens),**kwargs)
    def calibrate(target):
        def call(endpoint,body):
            return request('token-calibration',[dict(url=args.url,endpoint=endpoint,body=body)],30)[0]
        text,count,repeats=base.fitted_prompt(args.model,args.seed,waves,target,call,prompt_builder=retrieval_prompt)
        event(dict(event='shape',wave=waves,target=target,actual=count,repeats=repeats));return text
    try:
        # Start another full bounded wave until at least the requested sustained duration.
        # Final wave may finish up to ten minutes later; no idle padding counts as load.
        while time.monotonic()-started<args.seconds:
            check()
            if waves%2==0:
                gfx=True
                ssh('systemd-run','--user','--collect','--unit='+unit,'--property=RuntimeMaxSec=90min','--property=TimeoutStopSec=5','/usr/bin/python3',remote+'/x11_copy_workload.py')
            elif gfx:
                ssh('systemctl','--user','stop',unit);gfx=False
                unit='r9v-release-graphics-'+str(os.getpid())+'-'+str(waves+1)
            event(dict(event='graphics',active=gfx,unit=unit,wave=waves))
            long=calibrate(130816);medium=calibrate(rng.choice([32768,65536,98304]))
            decode=f'Run {args.seed}-{waves}. Write 500 numbered facts about rivers, with detailed explanations. Continue until the response limit.'
            jobs=[job(long,64),job(medium,64),job(decode,2048),job(decode+' Use different examples.',1024),job(decode,2048,cancel_chunks=rng.choice([1,3,17])),job(decode,2048,cancel_chunks=rng.choice([7,29,61])),job(decode,2048,disconnect_after=.25),job(decode,2048,disconnect_after=1.5)]
            order=list(range(8));rng.shuffle(order)
            event(dict(event='wave-order',wave=waves,order=order))
            results=request('eight-client-wave',[jobs[i] for i in order])
            for idx,r in zip(order,results):
                if idx<2:
                    target=130816 if idx==0 else None
                    actual=(r.get('usage') or {}).get('prompt_tokens',0)
                    if target and not target-96<=actual<=target+128:raise RuntimeError('Near128K usage mismatch')
                    if r['text_prefix'].strip()!='R9V-731':
                        error=dict(wave=waves,check='long-context-code',slot=idx,actual_tokens=actual,observed=r['text_prefix']);errors.append(error);event(dict(event='semantic-failure',**error))
            recovery=request('post-storm-recovery',[job('Compute 17 + 25. Reply only with the integer.',32)])[0]
            if recovery['text_prefix'].strip()!='42':
                error=dict(wave=waves,check='arithmetic',observed=recovery['text_prefix']);errors.append(error);event(dict(event='semantic-failure',**error))
            if gfx:
                assert ssh('systemctl','--user','is-active',unit).strip()=='active'
                log=ssh('journalctl','--user','-u',unit,'-n','100','--output=cat','--no-pager')
                assert '"event": "progress"' in log
                (output/f'graphics-{waves}.log').write_text(log)
            waves+=1
        if waves<8:raise RuntimeError('Fewer than eight complete waves')
    except BaseException as error:
        failure=repr(error);event(dict(event='failure',error=failure,traceback=traceback.format_exc()))
    finally:
        marker=campaign/'external-stop.json'
        if not marker.exists():base.save(marker,dict(time=time.time(),reason='Release stress '+(failure or 'completed')))
        cleanup_error=None
        try:
            if gfx:ssh('systemctl','--user','stop',unit)
        except Exception as error:cleanup_error=repr(error)
        base.save(output/'result.json',dict(stability_passed=not failure and not cleanup_error,release_passed=not failure and not cleanup_error and not errors,failure=failure,graphics_cleanup_error=cleanup_error,waves=waves,request_groups=groups,semantic_errors=errors,seconds=time.monotonic()-started,aftermath_verified=False))
    return 1 if failure or cleanup_error else 0

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seconds',type=int,default=3600);parser.add_argument('--seed',type=int,default=911731);parser.add_argument('--url',default='http://192.168.1.231:8004');parser.add_argument('--model',default='qwen3.8-flash-next');args=parser.parse_args()
    if not 3600<=args.seconds<=7200:parser.error('Duration must be3600..7200')
    def stop(*_):raise KeyboardInterrupt('Owned test interrupted')
    signal.signal(signal.SIGTERM,stop);sys.exit(run(args))
