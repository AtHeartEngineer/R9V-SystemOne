"""Bounded in-flight dispatch state for the blackbox controller."""
import json
import time


class LiveTrace:
    def __init__(self):
        self.pending = {}
        self.last_sequence = {}
        self.errors = []
        self.records = 0
        self.completions = 0
        self.last_completion_at = None
        self.named_pids = set()
        self.objects = {}

    def feed(self, line):
        if 'R9V_GPU ' not in line:
            return
        try:
            row = json.loads(line.split('R9V_GPU ', 1)[1])
            pid, seq = row['pid'], row['seq']
            previous = self.last_sequence.get(pid, 0)
            if seq != previous + 1:
                self.errors.append(dict(pid=pid, after=previous, next=seq))
            self.last_sequence[pid] = seq
            self.records += 1
            if row.get('error'):
                raise ValueError(row['error'])
            if row.get('type') == 'object_saved':
                self.objects[(pid,row['code_object'])] = row
            if row.get('type') == 'enqueue':
                if (pid,row.get('code_object')) not in self.objects:
                    raise ValueError('enqueue without captured code object')
                if row.get('name') not in (None, 'UNKNOWN'):
                    self.named_pids.add(pid)
                key = (pid, row['dispatch'])
                if len(self.pending) >= 20000:
                    raise ValueError('in-flight tracking budget exhausted')
                self.pending[key] = row
            elif row.get('type') == 'complete':
                self.pending.pop((pid, row['dispatch']), None)
                self.completions += 1
                self.last_completion_at = time.time()
        except (ValueError, KeyError) as error:
            self.errors.append(str(error))
        self.errors = self.errors[:20]

    def snapshot(self):
        return dict(time=time.time(), records=self.records, completions=self.completions,
                    named_pids=sorted(self.named_pids), code_objects=list(self.objects.values()),
                    last_completion_at=self.last_completion_at, errors=self.errors,
                    last_sequence=self.last_sequence, unfinished=list(self.pending.values()))
