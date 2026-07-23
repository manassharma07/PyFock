import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
basis_name = os.path.basename(script_dir)
basis_label = 'def2-SVP, SAO/spherical'
results_path = os.path.join(script_dir, f's22_{basis_name}_results.json')
log_path = os.path.join(script_dir, f's22_{basis_name}_benchmark.log')

with open(results_path) as handle:
    results = json.load(handle)
with open(log_path) as handle:
    log_text = handle.read()

blocks = re.split(r'(?=^Processing:)', log_text, flags=re.MULTILINE)
timings = {}
system_sizes = {}
for block in blocks:
    system_match = re.search(r'^Processing: .*[/\\]([^/\\]+)\.xyz$', block, re.MULTILINE)
    pyscf_match = re.search(r'^PySCF time:\s+([0-9.eE+-]+)', block, re.MULTILINE)
    pyfock_match = re.search(r'^Complete SCF\s+([0-9.eE+-]+)', block, re.MULTILINE)
    nbfs_match = re.search(r'^number of NR cGTOs =\s+(\d+)', block, re.MULTILINE)
    natoms_match = re.search(r'^Natoms\s+:\s*(\d+)', block, re.MULTILINE)
    if system_match and pyscf_match and pyfock_match:
        timings[system_match.group(1)] = (float(pyscf_match.group(1)), float(pyfock_match.group(1)))
    if system_match and nbfs_match and natoms_match:
        system_sizes[system_match.group(1)] = (int(natoms_match.group(1)), int(nbfs_match.group(1)))

for row in results:
    pyscf_time, pyfock_time = timings[row['system']]
    natoms, nbfs = system_sizes[row['system']]
    row['natoms'] = natoms
    row['nbfs'] = nbfs
    row['pyscf_wall_time_s'] = pyscf_time
    row['pyfock_wall_time_s'] = pyfock_time
    row['pyscf_wall_time_per_iteration_s'] = pyscf_time / row['pyscf_iterations']
    row['pyfock_wall_time_per_iteration_s'] = pyfock_time / row['pyfock_iterations']

reference = np.array([row['pyscf_energy_hartree'] for row in results])
predicted = np.array([row['pyfock_energy_hartree'] for row in results])
errors = predicted - reference
absolute_errors = np.abs(errors)
max_index = int(np.argmax(absolute_errors))
r_squared = 1.0 - np.sum(errors**2) / np.sum((reference-reference.mean())**2)

statistics = {
    'basis_set': basis_label,
    'n_systems': len(results),
    'rmse_hartree': float(np.sqrt(np.mean(errors**2))),
    'mae_hartree': float(np.mean(absolute_errors)),
    'mean_signed_error_hartree': float(np.mean(errors)),
    'median_absolute_error_hartree': float(np.median(absolute_errors)),
    'percentile_95_absolute_error_hartree': float(np.percentile(absolute_errors, 95)),
    'standard_deviation_signed_error_hartree': float(np.std(errors, ddof=1)),
    'max_absolute_deviation_hartree': float(absolute_errors[max_index]),
    'max_deviation_system': results[max_index]['system'],
    'r_squared': float(r_squared),
    'total_pyscf_wall_time_s': float(sum(row['pyscf_wall_time_s'] for row in results)),
    'total_pyfock_wall_time_s': float(sum(row['pyfock_wall_time_s'] for row in results)),
    'all_pyscf_converged': all(row['pyscf_converged'] for row in results),
    'all_pyfock_converged': all(row['pyfock_converged'] for row in results),
}

with open(os.path.join(script_dir, f's22_{basis_name}_analysis.json'), 'w') as handle:
    json.dump({'statistics': statistics, 'results': results}, handle, indent=2)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.8})
fig, (ax_parity, ax_error) = plt.subplots(1, 2, figsize=(7.2, 3.45), constrained_layout=True)
color = '#176B87'
accent = '#B23A48'

limits = (min(reference.min(), predicted.min()), max(reference.max(), predicted.max()))
padding = 0.025 * (limits[1]-limits[0])
limits = (limits[0]-padding, limits[1]+padding)
ax_parity.plot(limits, limits, color='#555555', linewidth=1.0, linestyle='--',
               label='Perfect agreement', zorder=1)
ax_parity.scatter(reference, predicted, s=31, color=color, edgecolor='white', linewidth=0.55,
                  label='S22 benchmark data points', zorder=2)
ax_parity.set(xlim=limits, ylim=limits, xlabel='PySCF total energy (Eₕ)',
              ylabel='PyFock total energy (Eₕ)', title=f'S22 energy parity ({basis_label})')
ax_parity.legend(loc='upper left', frameon=True, framealpha=0.96, fontsize=8)
ax_parity.text(0.97, 0.04,
               (f"R² = {statistics['r_squared']:.12f}\n"
                f"RMSE = {statistics['rmse_hartree']*1e6:.3f} μEₕ\n"
                f"MAE = {statistics['mae_hartree']*1e6:.3f} μEₕ"),
               transform=ax_parity.transAxes, ha='right', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                         edgecolor='#C7C7C7', alpha=0.96))

order = np.argsort(errors * 1e6)
sorted_errors = errors[order] * 1e6
sorted_names = [results[index]['system'] for index in order]
positions = np.arange(len(results))
ax_error.axhline(0, color='#555555', linewidth=0.9)
ax_error.axhline(statistics['mae_hartree']*1e6, color=accent, linewidth=0.8, linestyle=':')
ax_error.axhline(-statistics['mae_hartree']*1e6, color=accent, linewidth=0.8, linestyle=':')
ax_error.scatter(positions, sorted_errors, s=26, color=color, edgecolor='white', linewidth=0.45)
ax_error.set_xticks(positions)
ax_error.set_xticklabels(sorted_names, rotation=75, ha='right', fontsize=6.5)
ax_error.set(xlabel='S22 system (ordered by signed deviation)',
             ylabel='PyFock − PySCF (μEₕ)', title='Signed numerical deviations')

for axis in (ax_parity, ax_error):
    axis.grid(True, color='#D9E1E8', linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

fig.savefig(os.path.join(script_dir, f's22_{basis_name}_parity_plot.png'), dpi=600,
            bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(script_dir, f's22_{basis_name}_parity_plot.pdf'),
            bbox_inches='tight', facecolor='white')

fig_single, ax_single = plt.subplots(figsize=(4.25, 4.0), constrained_layout=True)
ax_single.plot(limits, limits, color='#555555', linewidth=1.0, linestyle='--',
               label='Perfect agreement', zorder=1)
ax_single.scatter(reference, predicted, s=34, color=color, edgecolor='white', linewidth=0.55,
                  label='S22 benchmark data points', zorder=2)
ax_single.set(xlim=limits, ylim=limits, xlabel='PySCF total energy (Eₕ)',
              ylabel='PyFock total energy (Eₕ)', title=f'S22 energy parity ({basis_label})')
ax_single.legend(loc='upper left', frameon=True, framealpha=0.96, fontsize=8)
ax_single.text(0.97, 0.04,
               (f"R² = {statistics['r_squared']:.12f}\n"
                f"RMSE = {statistics['rmse_hartree']*1e6:.3f} μEₕ\n"
                f"MAE = {statistics['mae_hartree']*1e6:.3f} μEₕ"),
               transform=ax_single.transAxes, ha='right', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
                         edgecolor='#C7C7C7', alpha=0.96))
ax_single.grid(True, color='#D9E1E8', linewidth=0.55, alpha=0.75)
ax_single.set_axisbelow(True)
for spine in ax_single.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)
    spine.set_color('#222222')
fig_single.savefig(os.path.join(script_dir, f's22_{basis_name}_parity_only.png'), dpi=600,
                   bbox_inches='tight', facecolor='white')
fig_single.savefig(os.path.join(script_dir, f's22_{basis_name}_parity_only.pdf'),
                   bbox_inches='tight', facecolor='white')
print(json.dumps(statistics, indent=2))
