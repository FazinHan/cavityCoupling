import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


filename = 'yig_t_0.100_lone.csv'
save_dir = 'just yig'
reqd_H = 1250 # Oe
num_of_minima = 5


directory = os.path.join(os.getcwd(),'data','yig_t_sweep_new')
full_save_dir = os.path.join(os.getcwd(),'images','cst', save_dir)

os.makedirs(full_save_dir, exist_ok=True)

data = pd.read_csv(os.path.join(directory,filename),index_col=0,dtype=float)

(idx,), = np.where(np.isclose(data.columns.values.astype(float), reqd_H))

min_idx = np.argmin(data.values[:,idx])

plt.plot(data.index.values.astype(float), data.values[:,idx],label=f'{reqd_H} Oe')
x = data.index.values.astype(float)
y = data.values[:, idx]

# Find local minima: y[i-1] > y[i] < y[i+1]
local_min_mask = np.r_[False, (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]), False]
minima_indices = np.where(local_min_mask)[0]

if minima_indices.size > 0:
    order = np.argsort(y[minima_indices])  # smallest values first
    k = min(num_of_minima, minima_indices.size)
    selected = minima_indices[order[:k]]
    for n, i_min in enumerate(selected):
        plt.annotate(
            f'{x[i_min]:.2f} GHz',
            xy=(x[i_min], y[i_min]),
            xytext=(x[i_min]+.1, y[i_min]-.05),
            arrowprops=dict(arrowstyle='->', lw=1.5),
        )
else:
    # Fallback: annotate global minimum if no local minima
    min_idx = np.argmin(y)
    plt.annotate(
        f'{x[min_idx]:.2f} GHz',
        xy=(x[min_idx], y[min_idx]),
        xytext=(x[min_idx], y[min_idx] - 0.15),
        arrowprops=dict(arrowstyle='->', lw=1.5),
    )
plt.xlabel('$f$ (GHz)')
plt.ylabel('$S_{21}$ (dB)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(full_save_dir, f'{reqd_H}oe.png'), dpi=300)
plt.show()
plt.close()