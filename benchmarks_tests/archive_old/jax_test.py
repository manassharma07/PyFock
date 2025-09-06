# again, this only works on startup!
from jax import config
config.update("jax_enable_x64", True)
import pylibxc
# import jax
import jax.numpy as jnp
from jax import grad, device_put, jit, random, jacobian, value_and_grad, vmap
import cupy as cp
from timeit import default_timer as timer
import numpy as np
from autograd import elementwise_grad, numpy as anp


@jit
def lda_x(rho):
    # Slater exchange functional (spin-paired).
    # Corresponds to the functional with the label LDA_X and ID 1 in Libxc.
    # Reference: Phys. Rev. 81, 385.
    
    pi34 = (3 / (4 * jnp.pi))**(1 / 3)
    f = -3 / 4 * (3 / (2 * jnp.pi))**(2 / 3)
    rs = pi34 * rho**(-1/3)
    ex = f / rs
    # ex = jnp.where(ex == jnp.nan, 0.0, ex)
    # ex = ex.at[jnp.isnan(ex)].set(0.0)
    # ex = jnp.nan_to_num(ex, copy=False, nan=0.0, posinf=None, neginf=None)
    return ex

@jit
def lda_c_vwn(rho):
    # Vosko-Wilk-Nusair parametrization of the correlation functional (spin-paired).
    # Corresponds to the functional with the label LDA_C_VWN and ID 7 in Libxc.
    # Reference: Phys. Rev. B 22, 3812.
    
    a = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
    pi34 = (3 / (4 * jnp.pi))**(1 / 3)
    rs = pi34 * jnp.power(rho, -1 / 3)
    q = jnp.sqrt(4 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = jnp.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = jnp.arctan(q / (2 * rs12 + b))
    ec = a * (jnp.log(rs / fx) + f1 * qx - f2 * (jnp.log((rs12 - x0)**2 / fx) + f3 * qx))
    return ec

funcx_id = 1
funcc_id = 7

# Create a LibXC object  
funcx = pylibxc.LibXCFunctional(funcx_id, "unpolarized")
funcc = pylibxc.LibXCFunctional(funcc_id, "unpolarized")
x_family_code = funcx.get_family()
c_family_code = funcc.get_family()

size = 10240
key = random.PRNGKey(0)
rho_jnp = random.uniform(key, (size, 1), dtype=jnp.float64)
rho_jnp = device_put(rho_jnp)
print(rho_jnp.dtype)

# rho_jnp[rho_jnp < 1e-12] = 0.0
# rho_jnp = rho_jnp.at[rho_jnp < 1e-12].set(0.0)

rho_np = np.asarray(rho_jnp)
print(rho_np.dtype)

rho_cp = cp.array(rho_np)
print(rho_cp.dtype)

# LDA slater exchange

start = timer()
for i in range(1000):
    e_jnp = lda_x(rho_jnp)
    # e_jnp = e_jnp.at[jnp.isnan(e_jnp)].set(0.0)
    # e_jnp = jnp.nan_to_num(e_jnp, copy=False, nan=0.0, posinf=None, neginf=None)
    # print(e_jnp)
duration = timer() - start
print('Jax duration   ', duration)


start = timer()
inp = {}
inp['rho'] = rho_np
for i in range(1000):
    retx = funcx.compute(inp)
    e_np = retx['zk']
    # print(e_np)
duration = timer() - start
print('Numpy duration   ', duration)

print(abs(e_np-np.asarray(e_jnp)).max())


# LDA VWN Correlation

start = timer()
for i in range(1000):
    e_jnp = lda_c_vwn(rho_jnp)
    # e_jnp = e_jnp.at[jnp.isnan(e_jnp)].set(0.0)
    e_jnp = jnp.nan_to_num(e_jnp, copy=False, nan=0.0, posinf=None, neginf=None)
    # print(e_jnp)
duration = timer() - start
print('Jax duration   ', duration)

start = timer()
inp = {}
inp['rho'] = rho_np
for i in range(1000):
    retc = funcc.compute(inp)
    e_np = retc['zk']
    # print(e_np)
duration = timer() - start
print('Numpy duration   ', duration)

print(abs(e_np-np.asarray(e_jnp)).max())

# LDA VWN Correlation gradient

def lda_c_vwn_autograd(rho):
    # Vosko-Wilk-Nusair parametrization of the correlation functional (spin-paired).
    # Corresponds to the functional with the label LDA_C_VWN and ID 7 in Libxc.
    # Reference: Phys. Rev. B 22, 3812.
    
    a = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
    pi34 = (3 / (4 * anp.pi))**(1 / 3)
    rs = pi34 * anp.power(rho, -1 / 3)
    q = anp.sqrt(4 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = anp.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = anp.arctan(q / (2 * rs12 + b))
    ec = a * (anp.log(rs / fx) + f1 * qx - f2 * (anp.log((rs12 - x0)**2 / fx) + f3 * qx))
    return ec

def lda_x_autograd(rho):
    # Slater exchange functional (spin-paired).
    # Corresponds to the functional with the label LDA_X and ID 1 in Libxc.
    # Reference: Phys. Rev. 81, 385.
    
    pi34 = (3 / (4 * anp.pi))**(1 / 3)
    f = -3 / 4 * (3 / (2 * anp.pi))**(2 / 3)
    rs = pi34 * rho**(-1/3)
    ex = f / rs
    # ex = jnp.where(ex == jnp.nan, 0.0, ex)
    # ex = ex.at[jnp.isnan(ex)].set(0.0)
    # ex = jnp.nan_to_num(ex, copy=False, nan=0.0, posinf=None, neginf=None)
    return ex

lda_c_vwn_grad = elementwise_grad(lda_x_autograd)
start = timer()
for i in range(1000):
    v_anp = lda_c_vwn_grad(rho_np)
    # e_jnp = e_jnp.at[jnp.isnan(e_jnp)].set(0.0)
    # v_jnp = jnp.nan_to_num(v_jnp, copy=False, nan=0.0, posinf=None, neginf=None)
    # print(e_jnp)
duration = timer() - start
print('Autograd duration   ', duration)
print(v_anp)


# lda_c_vwn_grad = vmap(grad(lda_c_vwn, argnums=0))
# lda_c_vwn_grad = jacobian(lda_c_vwn, 0)
# start = timer()
# for i in range(1000):
#     v_jnp = lda_c_vwn_grad([1.0,2.0,3.0])
#     # e_jnp = e_jnp.at[jnp.isnan(e_jnp)].set(0.0)
#     v_jnp = jnp.nan_to_num(e_jnp, copy=False, nan=0.0, posinf=None, neginf=None)
#     # print(e_jnp)
# duration = timer() - start
# print('Jax duration   ', duration)

start = timer()
inp = {}
inp['rho'] = rho_np
for i in range(1000):
    retc = funcx.compute(inp)
    v_np = retc['vrho']
    # print(e_np)
duration = timer() - start
print('Numpy duration   ', duration)
print(v_np)
print(abs(v_np-np.asarray(v_anp)).max())



