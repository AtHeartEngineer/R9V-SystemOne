# Exact-kernel timeout instrumentation, built but NOT loaded

This is a diagnostic alternative to rebuilding the full OGC kernel. It does not fix either the initiating GPU fault or the later host lockup. The original source patch remains unchanged in the parent incident directory.

The running package is kernel-core-7.1.5-ogc5.1.fc44.x86_64, source RPM kernel-core-7.1.5-ogc5.1.fc44.src.rpm. Installed kernel-devel reports kernel-7.1.5-ogc5.1.fc44.src.rpm. Full packaged source was not retrieved, so exact source equivalence to the OGC tag is NOT claimed. Instead, this module uses structure offsets extracted from the running vmlinux and amdgpu BTF, not guessed from the tag. Both raw BTF files and offsets.json are retained. The installed compiler matches the kernel's GCC 16.1.1 build identity.

At amdgpu_job_timedout entry the x86-64 first argument is drm_sched_job*. amdgpu_job.base offset is zero. It reads sched at +8, job.vmid +328, job.pasid +332. drm_gpu_scheduler.name is a pointer at +24; scheduler is embedded in amdgpu_ring at +120. Fence sync_seq and last_seq are at ring +32 and +36. Each read uses copy_from_kernel_nofault. No MMIO is read, no register is changed, no GPU is reset. A first-hit atomic latch bounds emission to two printk records. The callback returns zero to continue the original driver path.

Validation: `make -C /usr/src/kernels/7.1.5-ogc5.1.fc44.x86_64 M=/var/home/dylan/projects/inference/r9v/diagnostics/timeout-probe-7.1.5-ogc5 modules` completed under a 90-second timeout. See build.txt and modinfo.txt. BTF generation for the new module was skipped because vmlinux/pahole are unavailable; the probe does not consume its own BTF. The .ko is unsigned and would taint the running kernel as an external/unsigned module. Successful compilation does not validate attachment or callback execution.

## Concrete unresolved privilege boundary

`sudo -n true` on workstation fails: a password is required. The only listed passwordless workstation helper is r9v-sysrq-dump. Loading a kernel module or enabling tracefs probes requires privileges it does not grant. No insertion attempt, forced module load, privileged Docker, reboot, GPU reset or deliberate fault was performed.

An administrator can review this directory and the matching source directory on workstation, verify the hashes and live BTF, then use the supplied load-reviewed.sh. It checks exact kernel and BTF hashes and then inserts this one module. Do not load it if any identity check fails. The script does not dispatch inference, reboot or induce a fault. Removal is `rmmod r9v_timeout_probe` after capture and cessation of GPU work. A privilege grant alone does not validate attachment; successful load and kernel-log observation must be recorded before spending the remaining experiment allowance. Do not trigger a timeout merely to validate the probe.

Sources: [Kprobes](https://docs.kernel.org/trace/kprobes.html), [BTF and split BTF](https://docs.kernel.org/bpf/btf.html). Source timing is documented in the parent FINDINGS.md.
