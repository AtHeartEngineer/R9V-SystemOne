"""Single-writer bounded logs whose first-fault window cannot rotate away."""

import json
import os
import time
from pathlib import Path


class FaultWindow:
    def __init__(self, directory, limit=16 * 2**20, segments=4):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_sync = 0
        self.limit = limit
        self.segments = segments
        self.frozen = (self.directory / "first-fault").exists()

    def append(self, name, data):
        if self.frozen:
            return
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("log name must be a basename")
        if len(data) > self.limit:
            raise ValueError("record exceeds segment limit")
        path = self.directory / name
        if path.exists() and path.stat().st_size + len(data) > self.limit:
            for index in range(self.segments - 1, 0, -1):
                source = path if index == 1 else path.with_name(name + f".{index - 1}")
                if source.exists():
                    source.replace(path.with_name(name + f".{index}"))
        with path.open("ab", buffering=0) as stream:
            stream.write(data)
            if time.monotonic() - self.last_sync >= 0.25:
                os.fsync(stream.fileno())
                self.last_sync = time.monotonic()

    def flush(self):
        for path in self.directory.iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())

    def freeze(self, trigger):
        if self.frozen:
            return self.directory / "first-fault"
        # Latch before any filesystem work: even a failed snapshot stops rotation.
        self.frozen = True
        target = self.directory / "first-fault"
        target.mkdir()
        for path in self.directory.iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                path.replace(target / path.name)
        with (target / "trigger.json").open("x") as stream:
            json.dump(trigger, stream)
            stream.flush()
            os.fsync(stream.fileno())
        for directory in (target, self.directory):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return target
