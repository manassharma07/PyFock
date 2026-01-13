import platform
import psutil
import subprocess
import sys


def get_cpu_model():
    """
    Return CPU model information in a cross-platform manner.
    
    Retrieves the CPU model string using platform-specific methods:
    - Windows: Uses platform.processor()
    - macOS: Executes sysctl command to get brand string
    - Linux: Reads from /proc/cpuinfo
    - Other: Returns "Unknown CPU"
    
    Returns:
        str: The CPU model name/description, or an error message if retrieval fails.
        
    Examples:
        >>> cpu_model = get_cpu_model()
        >>> print(cpu_model)
        Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
        
    Note:
        On some systems, CPU model information may not be available or
        may require elevated privileges to access.
        
    Raises:
        No exceptions are raised directly, but any underlying system errors
        are caught and returned as descriptive error messages.
    """
    system = platform.system()

    try:
        if system == "Windows":
            return platform.processor()
        elif system == "Darwin":  # macOS
            command = ["sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(command).strip().decode()
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        else:
            return "Unknown CPU"
    except Exception as e:
        return f"Error getting CPU model: {e}"


def print_sys_info():
    """
    Print comprehensive system information to stdout.
    
    Displays detailed system specifications including:
    - Operating system name and version
    - System architecture (32-bit/64-bit)
    - CPU model and specifications
    - Number of physical CPU cores
    - Number of logical CPU threads
    - Total system memory in gigabytes
    
    The information is formatted in a human-readable format with
    labeled key-value pairs.
    
    Returns:
        None: This function only prints to stdout and returns nothing.
        
    Examples:
        >>> from pyfock import Utils
        >>> Utils.print_sys_info()
        Operating System: Linux 5.15.0
        System Type: 64bit
        CPU Model: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
        Number of Cores: 6
        Number of Threads: 12
        Memory (GB): 15.54
        
    Note:
        Some information may show as "Unavailable" if the system
        doesn't provide access to certain hardware details.
        
    Dependencies:
        Requires psutil, platform modules and the get_cpu_model() function.
    """
    print("\n============= System Information =============")
    # Get system specifications
    system_info = platform.uname()
    os = f"{system_info.system} {system_info.release}"
    system_type = platform.architecture()[0]
    num_cores = psutil.cpu_count(logical=False) or "Unavailable"
    num_threads = psutil.cpu_count(logical=True) or "Unavailable"
    memory = round(psutil.virtual_memory().total / (1024 ** 3), 2)  # in GB
    cpu_model = get_cpu_model()

    # Print system specifications
    print('Operating System:', os)
    print('System Type:', system_type)
    print('CPU Model:', cpu_model)
    print('Number of Cores:', num_cores)
    print('Number of Threads:', num_threads)
    print('Memory (GB):', memory)


if __name__ == "__main__":
    print_sys_info()
