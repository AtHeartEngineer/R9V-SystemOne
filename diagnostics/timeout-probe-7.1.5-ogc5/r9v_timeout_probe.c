// SPDX-License-Identifier: GPL-2.0
/* Diagnostic only. Offsets from the captured running kernel/module BTF.
 * Never read MMIO, change registers, reset a device or induce a timeout.
 */
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/uaccess.h>
#include <linux/utsname.h>
#include <linux/atomic.h>

static atomic_t captured = ATOMIC_INIT(0);

static int timeout_entry(struct kprobe *probe, struct pt_regs *regs)
{
	unsigned long job = regs->di, sched = 0, ring, name_ptr = 0;
	u32 pasid = 0, vmid = 0, signaled = 0, emitted = 0;
	char name[32] = {0};
	long errors = 0;

	if (atomic_cmpxchg(&captured, 0, 1))
		return 0;
	/* Log even if a subsequent nofault host-memory read fails. */
	pr_err("r9v amdgpu job timeout entry before IP dump\n");
	errors |= copy_from_kernel_nofault(&sched, (void *)(job + 8), sizeof(sched));
	errors |= copy_from_kernel_nofault(&vmid, (void *)(job + 328), sizeof(vmid));
	errors |= copy_from_kernel_nofault(&pasid, (void *)(job + 332), sizeof(pasid));
	if (sched) {
		ring = sched - 120;
		errors |= copy_from_kernel_nofault(&name_ptr, (void *)(sched + 24), sizeof(name_ptr));
		errors |= copy_from_kernel_nofault(&signaled, (void *)(ring + 36), sizeof(signaled));
		errors |= copy_from_kernel_nofault(&emitted, (void *)(ring + 32), sizeof(emitted));
		if (name_ptr)
			errors |= copy_from_kernel_nofault(name, (void *)name_ptr, sizeof(name) - 1);
	}
	pr_err("r9v amdgpu first timeout ring=%s pasid=%u vmid=%u signaled=%u emitted=%u read_error=%ld\n",
	       name, pasid, vmid, signaled, emitted, errors);
	return 0;
}
NOKPROBE_SYMBOL(timeout_entry);

static struct kprobe timeout_probe = {
	.symbol_name = "amdgpu_job_timedout",
	.pre_handler = timeout_entry,
};

static int __init r9v_probe_init(void)
{
	if (strcmp(utsname()->release, "7.1.5-ogc5.1.fc44.x86_64"))
		return -EINVAL;
	return register_kprobe(&timeout_probe);
}

static void __exit r9v_probe_exit(void)
{
	unregister_kprobe(&timeout_probe);
}

module_init(r9v_probe_init);
module_exit(r9v_probe_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("R9V first AMDGPU timeout host-memory identity capture; exact kernel only");
