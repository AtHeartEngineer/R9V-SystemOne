"""Incremental trace accounting. Callback receipt is not ordered GPU execution."""
import json


class TraceCheck:
    def __init__(self):
        self.last = {}
        self.gaps = []
        self.errors = []
        self.enqueued = {}
        self.completed = {}
        self.names = set()
        self.ready = set()
        self.records = 0

    def feed(self, line):
        if 'R9V_GPU ' not in line:
            return
        try:
            row = json.loads(line.split('R9V_GPU ', 1)[1])
        except ValueError:
            self.errors.append('malformed trace record')
            return
        self.records += 1
        if 'error' in row:
            self.errors.append(row['error'])
            return
        pid, seq = row['pid'], row['seq']
        if seq != self.last.get(pid, 0) + 1:
            self.gaps.append((pid, self.last.get(pid, 0), seq))
        self.last[pid] = seq
        kind = row.get('type')
        if kind == 'ready':
            self.ready.add(pid)
        if kind == 'enqueue':
            self.enqueued[(pid, row['dispatch'])] = row
            if row.get('name') not in ('UNKNOWN', None):
                self.names.add(row['name'])
        elif kind == 'complete':
            self.completed[(pid, row['dispatch'])] = row

    def result(self):
        return dict(records=self.records, ready=sorted(self.ready), gaps=self.gaps[:20],
                    errors=self.errors[:20], named_kernels=len(self.names),
                    enqueued=len(self.enqueued), completed=len(self.completed),
                    unfinished=[v for k, v in self.enqueued.items() if k not in self.completed])

    def validate_smoke(self):
        result = self.result()
        if (self.errors or self.gaps or len(self.ready) < 2 or not self.names
                or len(self.enqueued) < 20 or result['unfinished']
                or set(self.completed) - set(self.enqueued)):
            raise ValueError('GPU trace preflight failed: ' + json.dumps(result)[:3000])
        return result
