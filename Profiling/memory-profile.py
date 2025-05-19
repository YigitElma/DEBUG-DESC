#!/usr/bin/env python3
import subprocess
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import psutil


def monitor_ram(proc, interval, ram_usage, timestamps):
    """Sample system RAM until *proc* finishes."""
    while proc.poll() is None:  # child still running?
        info = psutil.virtual_memory()
        used_mb = (info.total - info.available) / 1024 / 1024
        ram_usage.append(used_mb)
        timestamps.append(time.time())
        time.sleep(interval)

    # keep watching for an extra second
    end = time.time() + 2.0
    while time.time() < end:
        info = psutil.virtual_memory()
        ram_usage.append((info.total - info.available) / 1024 / 1024)
        timestamps.append(time.time())
        time.sleep(interval)


def monitor_vram(proc, interval, vram_usage, timestamps):
    """Sample total GPU memory until *proc* finishes."""
    while proc.poll() is None:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        vram_usage.append(int(out.split()[0]))
        timestamps.append(time.time())
        time.sleep(interval)

    # keep watching for an extra second
    end = time.time() + 2.0
    while time.time() < end:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
        vram_usage.append(int(out.split()[0]))
        timestamps.append(time.time())
        time.sleep(interval)


if __name__ == "__main__":
    mode = "CPU"  # "CPU" or "GPU"
    res = 12
    maxiter = 2
    interval = 0.1  # seconds between samples

    mem_usage_gpu = []
    mem_usage_cpu = []
    ts_gpu = []
    ts_cpu = []

    # launch the script to be profiled
    child = subprocess.Popen(
        ["python", "test-script.py", f"{res}", f"{mode}", f"{maxiter}"]
    )

    # start the sampler thread
    sampler = threading.Thread(
        target=monitor_ram,
        args=(child, interval, mem_usage_cpu, ts_cpu),
        daemon=True,
    )
    if mode == "GPU":
        sampler2 = threading.Thread(
            target=monitor_vram,
            args=(child, interval, mem_usage_gpu, ts_gpu),
            daemon=True,
        )
        sampler2.start()
    sampler.start()

    # wait until the child exits, then join the sampler
    child.wait()
    sampler.join()
    if mode == "GPU":
        sampler2.join()

    branch = "master"
    # plotting
    mem_usage_cpu = np.asarray(mem_usage_cpu)
    mem_usage_cpu = mem_usage_cpu - min(mem_usage_cpu)
    times_cpu = np.asarray(ts_cpu) - ts_cpu[0]
    np.savetxt(f"{branch}_memory_cpu.txt", mem_usage_cpu)
    np.savetxt(f"{branch}_time_cpu.txt", times_cpu)

    if mode == "GPU":
        mem_usage_gpu = np.asarray(mem_usage_gpu)
        times_gpu = np.asarray(ts_gpu) - ts_gpu[0]
        np.savetxt(f"{branch}_memory_gpu.txt", mem_usage_gpu)
        np.savetxt(f"{branch}_time_gpu.txt", times_gpu)

    plt.figure(figsize=(15, 7))
    plt.plot(times_cpu, mem_usage_cpu, label="CPU", color="orange")
    if mode == "GPU":
        plt.plot(times_gpu, mem_usage_gpu, label="GPU", color="blue")
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Memory Usage (MB)", fontsize=20)
    plt.title(
        f"{mode} memory usage for test-script.py with resolution {res}, maxiter {maxiter}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"test-case-{mode}-res{res}-memory-{branch}.png", dpi=300)
    plt.show()
