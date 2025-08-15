# from omegaconf import OmegaConf

# cfg = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x3.yaml")
# print(OmegaConf.to_yaml(cfg))  # pretty print
# plain = OmegaConf.to_container(cfg)    # convert to plain dict/list for logging


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