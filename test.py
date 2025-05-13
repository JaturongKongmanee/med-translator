# import torch

# x = torch.rand(5, 3)
# print(x)



# print(torch.cuda.is_available())

# Here’s a Python script that uses NVIDIA's nvidia-smi command to retrieve detailed information about all NVIDIA GPUs on your system. This script parses the output and displays it in a readable format.

import subprocess

def get_nvidia_gpu_details():
    try:
        # Run the nvidia-smi command and capture the output
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print("Error: Unable to retrieve GPU details. Ensure NVIDIA drivers and nvidia-smi are installed.")
            print(result.stderr)
            return

        # Parse the output
        gpu_details = result.stdout.strip().split('\n')
        if not gpu_details:
            print("No NVIDIA GPUs detected.")
            return

        # Display GPU details
        print("NVIDIA GPU Details:")
        for gpu in gpu_details:
            index, name, total_mem, used_mem, free_mem, temp, utilization = gpu.split(', ')
            print(f"GPU {index}:")
            print(f"  Name: {name}")
            print(f"  Total Memory: {total_mem} MB")
            print(f"  Used Memory: {used_mem} MB")
            print(f"  Free Memory: {free_mem} MB")
            print(f"  Temperature: {temp} °C")
            print(f"  Utilization: {utilization} %")
            print("-" * 30)

    except FileNotFoundError:
        print("Error: nvidia-smi command not found. Please ensure NVIDIA drivers are installed and nvidia-smi is in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_nvidia_gpu_details()

# Key Notes:
# Dependencies: Ensure that the NVIDIA drivers and nvidia-smi are installed and accessible in your system's PATH.
# Cross-Platform: This script works on systems where nvidia-smi is supported (e.g., Linux, Windows).
# Permissions: You may need administrative privileges to run nvidia-smi on some systems.

# Run this script in a Python environment, and it will display all the relevant GPU details in a user-friendly format.

