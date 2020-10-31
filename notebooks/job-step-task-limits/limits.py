#!/usr/bin/env python3
import os
import glob
import time

slurm = {
    k[len('SLURM_'):]: v 
    for k, v in os.environ.items() 
    if k.startswith('SLURM_')
}

# Job info (printed by task 0 in step 0)
if int(slurm['STEP_ID']) == 0 and int(slurm['PROCID']) == 0:
    print(f"Job {slurm['JOB_ID']}")
    print(
        f"  Submit host: {slurm['SUBMIT_HOST']}",
        f"  Nodes      : {slurm['JOB_NODELIST']}",
        f"  Num nodes  : {slurm['JOB_NUM_NODES']}",
        sep='\n',
        end='\n\n',
    )
    
# Step info (printed by task 0 in each step)
if int(slurm['PROCID']) == 0:
    print(f"Step {slurm['STEP_ID']}")
    print(
        f"  Nodes      : {slurm['STEP_NODELIST']}",
        f"  Num nodes  : {slurm['STEP_NUM_NODES']}",
        f"  Num tasks  : {slurm['STEP_NUM_TASKS']}",
        sep='\n',
        end='\n\n',
    )
    
# Sleep to avoid overlaps in the log file
time.sleep(int(slurm['PROCID']))

# Get cgroup of this process
# E.g. /slurm/uid_012345/job_001122/step_0/task_0
with open(f'/proc/{os.getpid()}/cgroup') as f:
#     cgroup = next(l.strip() for l in f if ':memory' in l).split(':')[2]
    for l in f:
        if ':memory' in l:
            cgroup = l.strip().split(':')[2]    
            *_, uid, job, step, task = cgroup.split('/')
            uid = int(uid[len('uid_'):])
            job = int(job[len('job_'):])
            step = int(step[len('step_'):])
            task = int(task[len('task_'):])
            break
    else:
        raise RuntimeError('Unable to parse cgroup info')

# CPU quota of all tasks in this step
# https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-cpuacct
cpu_shares_tot = 0
for p in glob.glob(f'/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{uid}/job_{job}/step_{step}/task_*/cpu.shares'):
    with open(p) as f:
        cpu_shares_tot += int(f.readline())
            
# CPU quota of this task (as a fraction of the step quota)
with open(f'/sys/fs/cgroup/cpu,cpuacct/slurm/uid_{uid}/job_{job}/step_{step}/task_{task}/cpu.shares') as f:
    cpu_shares_task = int(f.readline())
    
# CPU cores bound to this step (shared among tasks)
# https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-cpuset
with open(f'/sys/fs/cgroup/cpuset/slurm/uid_{uid}/job_{job}/step_{step}/cpuset.cpus') as f:
    cpu_cores = f.readline().strip()
    
# RAM 
# Soft/hard limits in bytes for this step/task
# https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-memory
with open(f'/sys/fs/cgroup/memory/slurm/uid_{uid}/job_{job}/step_{step}/memory.limit_in_bytes') as f:
    mem_step_hard = float(f.readline())
with open(f'/sys/fs/cgroup/memory/slurm/uid_{uid}/job_{job}/step_{step}/memory.soft_limit_in_bytes') as f:
    mem_step_soft = float(f.readline())
with open(f'/sys/fs/cgroup/memory/slurm/uid_{uid}/job_{job}/step_{step}/task_{task}/memory.limit_in_bytes') as f:
    mem_task_hard = float(f.readline())
with open(f'/sys/fs/cgroup/memory/slurm/uid_{uid}/job_{job}/step_{step}/task_{task}/memory.soft_limit_in_bytes') as f:
    mem_task_soft = float(f.readline())
    
print(
    f"Task {slurm['PROCID']}",
    f"  Hostname: {os.environ['SLURMD_NODENAME']}",

    # process ID of the task being started
    f"  Task PID: {slurm['TASK_PID']:<6}",
    
    # step-wise task ID, i.e. MPI rank
    f"  Task id : {slurm['PROCID']:<6}",
    
    # node local task ID for the process within this step
    f"  Local id: {slurm['LOCALID']:<6}",
    
    f"  Task control group : {cgroup}",
    
    # CPU cores shared among the tasks in this step on this machine
    f"  CPU cores  per step: {cpu_cores:<10}",
    
    # Percentage of the above cores that the task should use
    f"  CPU shares per task: {cpu_shares_task/cpu_shares_tot:<10.2%}",
    f"  RAM limits per step: {mem_step_soft / 2**30:.2f} GB (hard {mem_step_hard / 2**30:.2f} GB)",
    f"  RAM limits per task: {mem_task_soft / 2**30:.2f} GB (hard {mem_task_hard / 2**30:.2f} GB)",
    sep='\n',
    end='\n\n',
)
