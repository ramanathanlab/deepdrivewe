"""Code to analyze siulation speeds."""

from __future__ import annotations

import fnmatch
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

runs = ['ntl9-v1_1node', 'ntl9-v1_2node', 'ntl9-v1_4node', 'ntl9-v1_8node']
nodes = [1, 2, 4, 8]
ppn = 12
benchmark = 10000
agg_simspeed = []
for r, n in zip(runs, nodes):
    """ """
    # Define the base directory and pattern

    base_dir = f'runs/{r}/simulation'
    pattern = '*.log'  # Example pattern for text files
    speed_ns_day = []
    # Traverse the directory tree
    for root, dirs, files in tqdm(os.walk(base_dir)):
        if dirs == '':
            print(dirs)
        for filename in fnmatch.filter(files, pattern):
            file_path = os.path.join(root, filename)
            # print(f'Found file: {file_path}')
            try:
                df = pd.read_csv(file_path)
                mn_speed = np.mean(df['Speed (ns/day)'].to_numpy())
                if mn_speed < benchmark:
                    speed_ns_day.append(mn_speed)
                else:
                    continue
            except Exception:
                continue
            del df
    agg_simspeed.append(np.mean(speed_ns_day) * ppn * n)

plt.scatter(nodes, agg_simspeed)
plt.plot(nodes, agg_simspeed)
plt.xlabel('Number of PVC nodes (12 instances/node)')
plt.ylabel('Aggregate simulation speed (ns/day)')
plt.savefig('average_speed_scaling.png', bbox_inches='tight', dpi=300)
plt.close()
print(agg_simspeed)
