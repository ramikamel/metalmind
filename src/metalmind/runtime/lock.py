"""Serialize GPU measurements across local MetalMind processes."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import tempfile
import time

@contextmanager
def gpu_lock(timeout=30):
    path=Path(tempfile.gettempdir())/f"metalmind-gpu-{os.getuid()}.lock"
    with path.open("a+") as f:
        deadline=time.monotonic()+timeout
        while True:
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>=deadline: raise TimeoutError("GPU evaluation lock timed out")
                time.sleep(.05)
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)
