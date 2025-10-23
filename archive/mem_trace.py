import psutil
import threading
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import gc

class MemoryTrace:
    
    def __enter__(self):
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.begin = torch.cuda.memory_allocated()
            print(f"begin with gpu memory occupied: {self.begin}")
        
        self.cpu_peaks = []
        self.peak_monitoring = True
        self.process = psutil.Process()
        
        self.peak_monitoring = True
        
        peak_monitor_thread = threading.Thread(target=self.peak_memory_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start() # no cleanup or join process because the thread is running in damon.
        return self
        
    def cpu_mem_used(self):
        return self.process.memory_info().rss

    def peak_memory_func(self):
        
        self.cpu_peak = -1
        
        while True:
            self.cpu_peak = max(self.cpu_peak, self.cpu_mem_used())
            self.cpu_peaks += [self.cpu_peak]
            if not self.peak_monitoring:
                break
            
    def b2mb(self, b):
        return float(b / 2**20)
    
    def __exit__(self, *exc):
        self.peak_monitoring = False
        if torch.cuda.is_available():
            torch.cuda.synchronize() # synchornize torch memory stats
            self.end = torch.cuda.memory_allocated()
            self.peak = torch.cuda.max_memory_allocated()
            self.used = self.b2mb(self.end - self.begin)
            self.peaked = self.b2mb(self.peak - self.begin)
            
        gc.collect()
        self.cpu_peak = self.b2mb(self.cpu_peak)
    
device = "cuda:0"
_ = torch.randn(10, 50).to(device)

time.sleep(1)

with MemoryTrace() as trace:
    matrix = np.random.randn(10, 50)
    matrix_gpu = torch.randn(10, 50).to(device)
    print("Current GPU mem:", torch.cuda.memory_allocated())

print(f"cpu memory peak: {trace.cpu_peak}")
print(f"gpu memory used during tracing: {trace.used}")
print(f"gpu memory peaked during traacing: {trace.peaked}")

x = [i for i in range(len(trace.cpu_peaks))]
plt.plot(x, trace.cpu_peaks)
plt.show()
plt.savefig('cpu_peaks.png')