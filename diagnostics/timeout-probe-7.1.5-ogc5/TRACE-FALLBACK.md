# Unsigned-module fallback

The workstation rejected `insmod` with `Key was rejected by service`; its
kernel journal says `Loading of unsigned module is rejected`. Secure Boot is
enabled, module signature enforcement is Y, and lockdown currently shows
`[none]`. No signing credentials were found in the standard akmods private-key
location. Retrying sudo or SSH credentials will not resolve module signing.

`trace-timeout.py` uses the kernel's compiled-in KPROBE_EVENTS interface. It
requires root, but does not load a module or change boot/security settings.
Its kernel release, architecture, and live BTF hash guards match the reviewed
module. The same offsets read ordinary host memory, never GPU MMIO.

Run on the workstation:

```bash
sudo python3 /var/home/dylan/projects/inference/r9v/diagnostics/timeout-probe-7.1.5-ogc5/trace-timeout.py --install
```

This installs a root-owned recorder and systemd service, starts it, and waits
for attach readiness. It does not start inference or enable itself at boot.
The installer can repair the exact original failed recorder or reinstall the
current version while the service is inactive/failed. Running services,
unknown program versions, modified units, and symlinks are refused. Existing
trace instances/events are still refused instead of overwritten.
Inspect errors with `journalctl -u r9v-timeout-trace.service -n 30 --no-pager`.

The isolated trace instance records only `amdgpu_job_timedout` entry, with job
and scheduler pointers, ring name, PASID, VMID, and signaled/emitted counters.
Its buffer is 64 KiB per CPU, does not overwrite old records, and wakes the
reader without waiting for a buffer fill threshold. The first delivered
timeout record is sent to the existing blackbox UDP receiver on port 6667,
also written to the service journal, then tracing is disabled. Readiness and
30-second heartbeat messages use the same channel. The existing blackbox
receiver fsyncs received messages. No fault was induced to validate this.

This is weaker than the module's direct printk/netconsole path: the reader
must be scheduled before it can forward the event, UDP can lose a packet,
and a hard CPU lockup can prevent forwarding. A first delivered record is
not an atomic global first-timeout latch across CPUs. Failed kprobe memory
reads can yield zero or a fault marker; zero does not prove a real field value.
The reported PASID is a GPU address-space ID, not a host PID. Correlate it
with the existing process/cgroup/worker journals. Existing netconsole capture
remains essential. Do not claim this identifies the initiating fault yet.

Validated before root attach: Python syntax, live kernel/BTF guards, systemd
unit syntax, and a CPU-only UDP test received and persisted on blackbox
(`trace-fallback-20260909`). Kernel acceptance of this event definition and
service readiness still require the sudo step. Afterward verify a matching
`source=r9v-timeout-trace, kind=ready` record on blackbox before any GPU test.
The campaign controller's old module-loaded gate must be reviewed explicitly;
this fallback must not be represented as that module being loaded.

Stop and detach cleanly:

```bash
sudo systemctl stop r9v-timeout-trace.service
```

The service removes only its own trace event and instance on normal stop.
If killed forcibly, inspect residual state before removing it. No automatic
service restart or post-fault rearming occurs. The cumulative at-most-one
additional-host-crash budget and the prohibition on deliberate panic tests
remain unchanged.

Interface reference: https://docs.kernel.org/trace/kprobetrace.html

## September 9 attach failure repair

The first trace service attempt failed at `Path.open("a")`, before writing
the definition. CPython append-mode FileIO seeks to EOF at open; the tracefs
control file uses `seq_lseek`, which rejects SEEK_END with EINVAL. The writer
now uses `os.open(O_WRONLY | O_CLOEXEC)` and one `os.write`, with no seek,
creation, or truncation. Using Python `"w"` here would be unsafe because
O_TRUNC on kprobe_events removes existing probes.

An unprivileged regression experiment intercepted SEEK_END for a temporary
control file and returned EINVAL: the original append-mode path failed during
open, and the corrected writer succeeded under the same condition. Installer
checks passed for known failed-version repair, active-service refusal,
unknown-file/unit refusal, symlink refusal, and atomic replacement. The live
kernel/BTF guards and inspection of the installed failed version also passed.
These tests do not prove live kernel event acceptance. Rerun the same sudo
`--install` command above to deploy the corrected recorder and attempt attach.

Sources: https://github.com/python/cpython/blob/v3.14.0/Modules/_io/fileio.c
and https://github.com/torvalds/linux/blob/master/kernel/trace/trace_kprobe.c
