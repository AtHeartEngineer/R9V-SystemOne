"""Keep stdout/stderr byte fragments separate until complete lines reach the collector."""
import os,selectors,subprocess,time

def run(args, timeout=None):
    proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    sel=selectors.DefaultSelector()
    for pipe in (proc.stdout,proc.stderr):sel.register(pipe,selectors.EVENT_READ,bytearray())
    deadline=time.monotonic()+timeout if timeout else None
    try:
        while sel.get_map():
            if deadline and time.monotonic()>deadline:raise subprocess.TimeoutExpired(args,timeout)
            for key,_ in sel.select(.2):
                chunk=os.read(key.fd,65536)
                pending=key.data
                if not chunk:
                    if pending:os.write(1,bytes(pending)+b'\n')
                    sel.unregister(key.fileobj);key.fileobj.close();continue
                pending.extend(chunk)
                while b'\n' in pending:
                    line,_,rest=pending.partition(b'\n');pending[:]=rest
                    data=memoryview(bytes(line)+b'\n')
                    while data:
                        wrote=os.write(1,data)
                        if wrote<=0:raise RuntimeError('output write failed')
                        data=data[wrote:]
                if len(pending)>1048576:raise RuntimeError('oversized source log line')
        code=proc.wait(timeout=5)
        if code:raise subprocess.CalledProcessError(code,args)
        return code
    finally:
        sel.close()
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)

if __name__=='__main__':
    import sys
    run(sys.argv[1:])
