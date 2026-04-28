import torch
import time
import psutil
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
USE_CUDA = 0  # 假设您现在纯测 CPU

class ProfiledAttack:
    def __init__(self, inner_attack):
        self.attack = inner_attack
        self.logs = []
        self.process = psutil.Process(os.getpid()) # 获取当前操作系统的物理进程

    def edit(self, text, *args, **kwargs):
        # ---- 强制垃圾回收，清理之前的残留，获取最干净的系统内存基线 ----
        import gc
        gc.collect()
        
        # 记录攻击前的物理内存 (RSS)
        cpu_mem_before = self.process.memory_info().rss 
        
        # 启动一个后台线程/或直接记录峰值（为了轻量，我们简单记录首尾和可能的最大值）
        # 如果要极度精确，可以用 memory_profiler，但在单次 edit 中，直接比对差值通常能反映算力开销。
        
        t0 = time.perf_counter()
        result = self.attack.edit(text, *args, **kwargs)
        t1 = time.perf_counter()

        # 记录攻击后的物理内存
        cpu_mem_after = self.process.memory_info().rss
        
        # 计算真实的 CPU 物理内存增量 (转换为 MB)
        # 注意：这里测的是净增量。如果攻击中有瞬间峰值被释放，可以用多线程采样，但通常对于评估轻量化，净增量或PyTorch的追踪就足够了。
        cpu_overhead_mb = abs(cpu_mem_after - cpu_mem_before) / (1024 ** 2)

        self.logs.append({
            'time_s':      t1 - t0,
            'cpu_peak_mb': cpu_overhead_mb,
        })
        return result

    def print_summary(self):
        n = len(self.logs)
        if n == 0: return
        avg_t   = sum(x['time_s'] for x in self.logs) / n
        avg_cpu = sum(x['cpu_peak_mb'] for x in self.logs) / n
        max_t   = max(x['time_s'] for x in self.logs)
        max_cpu = max(x['cpu_peak_mb'] for x in self.logs)

        print("\n" + "=" * 58)
        print("  UPV-Gradient-S  单样本 attack.edit() [OS级物理内存] 统计")
        print("=" * 58)
        print(f"  {'攻击耗时 (s)':<20} {avg_t:>12.4f} {max_t:>12.4f}")
        print(f"  {'CPU真实物理增量 (MB)':<20} {avg_cpu:>12.2f} {max_cpu:>12.2f}")
        print("=" * 58 + "\n")