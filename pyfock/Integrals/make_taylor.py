# The following source code was obtained from https://github.com/peter-reinholdt/pyboys under the BSD-3 license

#!/usr/bin/env python
import sympy
import mpmath
import numpy as np

max_order        = 16
max_angular      = 10
expansion_points = np.arange(-50, 50, dtype=np.float64)
offset           = - int(expansion_points[0])
num_points       = len(expansion_points)


z,z0 = sympy.symbols('z z0', assume='real')
a    = sympy.symbols('a', assume='integer')
f = sympy.Function('fun')


hyp = sympy.hyper([a],[a+1], z)
gen = sympy.series(hyp, z, x0=z0, n=None)
exp = 0
for i in range(max_order):
    # nextgen = next(gen).subs(sympy.hyper((a+i,), (a+i+1,), z0), 'table[ai+{},{}]'.format(i,'zi'))
    nextgen = next(gen).subs(sympy.hyper((a+i,), (a+i+1,), z0), 'table(ai+{},{})'.format(i,'zi')) # Fixed by bing chat
    print(nextgen)
    exp += nextgen


with open("taylor.py", "w") as f:
    f.write("import numpy as np\n")
    f.write("from numba import jit\n")
    f.write("\n\n")
    f.write("table = np.array([\n")
    for i in range(max_angular+max_order):
        print(i)
        f.write("{},\n".format([float(mpmath.hyp1f1(i+0.5, i+1.5, x)) for x in expansion_points]))
    f.write("])\n")
    f.write("\n\n")
    f.write("@jit(nopython=True, cache=True)\n")
    f.write("def taylor(a,z):\n")
    f.write("    z0 = int(np.round(z))\n")
    f.write("    zi = z0 + {}\n".format(offset))
    f.write("    ai = a\n")
    f.write("    return {}\n".format(exp.subs(a,a+0.5).evalf()))