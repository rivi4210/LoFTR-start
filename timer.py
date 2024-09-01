import torch
import time
# פונקציות למדידת זמן GPU
class Timer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0
    def start_timer(self):
        self.start.record()
    def end_timer(self):
        self.end.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start.elapsed_time(self.end)
    def get_elapsed_time(self):
        return self.elapsed_time / 1000  # Convert milliseconds to seconds
# פונקציות למדידת שימוש בזיכרון GPU
def get_memory_usage():
    return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
     #
def get_memory_max():
    return torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert bytes to MB
def print_memory_usage():
    print(f"Memory allocated: {get_memory_usage():.2f} MB memory max allocated:{get_memory_max():.2f}")