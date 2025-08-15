'''
This script shows how to control a running Python process with Unix signals,
without stopping it.

Key idea
--------
Unix lets you send "signals" to a running process. The process can register
handlers for specific signals. When a signal arrives, the OS interrupts normal
execution, runs the handler function, and then the program continues.

Why this is useful
------------------
- You can force a checkpoint at any time (for example before a queue time limit).
- You can debug a live, possibly stuck process without killing it.

What this script does
---------------------
- It registers two handlers:

  1) SIGUSR1  -> save_ckpt()
     When the process receives SIGUSR1, save_ckpt() runs. Here it writes a
     dummy checkpoint file "last.ckpt". In a real training job you would call
     trainer.save_checkpoint(...) or your own save code.

  2) SIGUSR2  -> debug_here()
     When the process receives SIGUSR2, debug_here() runs and drops into the
     Python debugger (pdb.set_trace()) at that exact point in the live process.

- It then runs an infinite loop that prints a counter every 2 seconds to simulate
  a long job.

How to use from another terminal
--------------------------------
1) Start this script:
   $ python play.py
   It will print the process PID.

2) From another terminal (same machine), send signals:

   Save a checkpoint immediately (non-interactive, safe in batch jobs):
   $ kill -USR1 <PID>

   Enter the debugger inside the live process (requires interactive terminal):
   $ kill -USR2 <PID>

Notes for Slurm
---------------
- For a job submitted with sbatch, you can send SIGUSR1 without logging into the
  node:
    $ scancel --signal=USR1 <jobid>
  That will trigger save_ckpt() in the running job.

- SIGUSR2 starts an interactive debugger. That needs a terminal attached to the
  job. If you send SIGUSR2 to a non-interactive sbatch job, it will block at the
  debugger prompt that you cannot see. Use srun --pty or an interactive session
  if you want to use SIGUSR2.

Safety and behavior
-------------------
- Handlers only run on signal receipt. During normal execution they do nothing.
- After a handler returns, the program continues from where it was interrupted.
- In multi-GPU Lightning jobs, you usually guard the handler to run only on rank 0.

Common commands
---------------
- Find PID:
  $ ps -u $USER | grep play.py

- Send signals:
  $ kill -USR1 <PID>   # save now
  $ kill -USR2 <PID>   # debugger now

- Slurm:
  $ scancel --signal=USR1 <jobid>   # safe on sbatch jobs
  (avoid SIGUSR2 on non-interactive sbatch runs)
'''



import time
import os
import signal
import pdb

# Fake "checkpoint save"
def save_ckpt(*args, **kwargs):
    print(f"[PID {os.getpid()}] Saving checkpoint now...")
    # Simulate writing to disk
    with open("last.ckpt", "w") as f:
        f.write("fake checkpoint data\n")
    print(f"[PID {os.getpid()}] Checkpoint saved!")

# Drop into interactive debugger
def debug_here(*args, **kwargs):
    print(f"[PID {os.getpid()}] Dropping into debugger...")
    pdb.set_trace()  # interactive terminal debugging

# Register handlers
signal.signal(signal.SIGUSR1, save_ckpt)
signal.signal(signal.SIGUSR2, debug_here)

print(f"Process PID = {os.getpid()}")
print("Send signals from another terminal, e.g.:")
print(f"  kill -USR1 {os.getpid()}   # save checkpoint immediately")
print(f"  kill -USR2 {os.getpid()}   # drop into debugger")

# Simulate a long-running job
counter = 0
while True:
    print(f"[PID {os.getpid()}] Running... counter={counter}")
    time.sleep(2)
    counter += 1
