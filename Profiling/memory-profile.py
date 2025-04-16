#!/usr/bin/env python3
import subprocess
import time
import threading
import matplotlib.pyplot as plt
import numpy as np
import psutil


def monitor_vram(duration, interval, vram_usage_list, timestamps):
    end_time = time.time() + duration
    while time.time() < end_time:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
        )
        output = result.stdout.decode("utf-8").strip()
        vram_usage = int(output.split()[0])
        vram_usage_list.append(vram_usage)
        timestamps.append(time.time())
        time.sleep(interval)


def monitor_ram(duration, interval, ram_usage_list, timestamps):
    """Monitor system RAM usage over time.

    Parameters:
        duration (float): Total duration to monitor (in seconds).
        interval (float): Time interval between measurements (in seconds).
        ram_usage_list (list): List to store RAM usage values in MB.
        timestamps (list): List to store timestamps.
    """
    end_time = time.time() + duration
    while time.time() < end_time:
        ram_info = psutil.virtual_memory()
        ram_used_mb = (ram_info.total - ram_info.available) / 1024 / 1024  # in MB
        ram_usage_list.append(ram_used_mb)
        timestamps.append(time.time())
        time.sleep(interval)


if __name__ == "__main__":

    mode = "CPU"

    duration = 70  # duration to monitor in seconds
    interval = 0.1  # interval between checks in seconds
    vram_usage_list = []
    timestamps = []

    # create threads for monitoring VRAM and running GPU code
    target = monitor_ram if mode == "CPU" else monitor_vram
    vram_thread = threading.Thread(
        target=target, args=(duration, interval, vram_usage_list, timestamps)
    )

    # start the threads
    vram_thread.start()
    subprocess.Popen(["python", "test-script.py"])

    # wait for the thread to finish
    vram_thread.join()

    # # write the VRAM usage to a file
    # with open("vram_usage_with1688_delQR.txt", "w") as file:
    #     for usage in vram_usage_list:
    #         file.write(f"{usage}\n")

    plt.figure(figsize=(15, 7))
    # plot the VRAM usage
    times = [t - timestamps[0] for t in timestamps]

    vram_usage_list = np.array(vram_usage_list)
    times = np.array(times)

    # this is kind of ad-hoc way of clipping the intended section
    idx = np.where(vram_usage_list > 1000)
    idx = np.arange(len(vram_usage_list))

    plt.plot(times[idx], vram_usage_list[idx])
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Memory Usage (MiB)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f"{mode} Memory Usage Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"test-case-res10-{mode}-with1688.png", dpi=1000)
    plt.show()
