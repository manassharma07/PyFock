# When performing benchmarks, it might be useful to print out the specifications
# of the system, the calculation is being run on

from pyfock import Utils

Utils.print_sys_info()

print(Utils.get_cpu_model())