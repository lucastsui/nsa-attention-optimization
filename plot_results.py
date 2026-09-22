"""Render standalone scientific figures from checked-in measurements (matplotlib 3.10.6)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'svg.fonttype':'none', 'savefig.dpi':180})
summary=json.loads((HERE/'results/study_summary.json').read_text())
names=[f'n{n}_{p}_s29' for n in (4096,8192,16384) for p in ('random','recent','nsa')]
labels=[f'{n//1024}K\n{p}' for n in (4096,8192,16384) for p in ('scattered','recent','selector')]
fig,ax=plt.subplots(figsize=(11.4,5.0))
for i,s in enumerate(summary['sessions']):
    lookup={r['case']:r for r in s['comparisons'] if r['control']=='fla_tuned'}
    values=np.array([lookup[n]['speedup'] for n in names])
    low=np.array([min(lookup[n]['round_speedups']+[lookup[n]['speedup']]) for n in names])
    high=np.array([max(lookup[n]['round_speedups']+[lookup[n]['speedup']]) for n in names])
    ax.errorbar(np.arange(9)+(i-.5)*.17, (values-1)*100,
                yerr=np.stack((values-low,high-values))*100, fmt='o',capsize=3,
                color=('#1565a8','#db7730')[i],label=f'Session {i+1}',markersize=6)
ax.axhline(0,color='#555',linewidth=.8)
ax.set_xticks(range(9),labels)
ax.set_ylabel('Throughput gain over tuned original (%)')
ax.set_title('Delayed V loading improves selected-attention throughput',loc='left',pad=18,weight='bold')
ax.grid(axis='y',alpha=.2)
ax.legend(frameon=False,loc='lower left')
fig.text(.09,.015,'RTX 5090 Laptop · BF16 · G16 · Hkv4 · D128 · block64 · top16\nDots: ratio of session medians. Whiskers: range of seven round ratios, not a confidence interval. Q/K/V are synthetic.',fontsize=9,color='#444')
fig.subplots_adjust(bottom=.22,top=.88,left=.09,right=.98)
for ext in ('png','svg'): fig.savefig(OUT/f'throughput.{ext}')
plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(10.6,3.9))
labels=['Original\n4 warps','Original\n8 warps','Delayed V\n4 warps']
profile=json.loads((HERE/'results/experiment_summary.json').read_text())
arms=('fla','fla_w8','late_v_w4')
series=([profile['target_resources'][p]['shared_bytes']/1024 for p in arms],
        [profile['target_resources'][p]['registers_per_thread'] for p in arms],
        [float(profile['controlled_profile'][p]['sm__warps_active.avg.pct_of_peak_sustained_active']['value']) for p in arms])
for ax,values,title,ylabel in zip(axes,series,
                                  ('Shared memory','Registers per thread','Achieved occupancy'),('KiB','Registers','Percent')):
    bars=ax.bar(labels,values,color=['#8295a8','#b4bec9','#1565a8'],width=.6)
    ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ax.set_title(title,loc='left',weight='bold')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,max(values)*1.22)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
fig.text(.04,.025,'Nsight Compute, N8192 scattered workload. Shared memory excludes 1 KiB driver allocation per block.\nEight warps raised occupancy but did not improve ordinary timing; more occupancy alone is not the explanation.',fontsize=9,color='#444')
fig.subplots_adjust(bottom=.24,wspace=.35,left=.06,right=.98,top=.85)
for ext in ('png','svg'): fig.savefig(OUT/f'resources.{ext}')
plt.close(fig)
print('Wrote figures/throughput.{png,svg} and figures/resources.{png,svg}')
