import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotter_t_sweep_p_w import csv_files

if __name__ == "__main__":
    

    os.makedirs("results\\yig_t_sweep_plots\\peaks_widths", exist_ok=True)

    for file in csv_files:
        # plotter()

        pivot_table = pd.read_csv(file)

        equality_idx = np.where(pivot_table['xc1'] == pivot_table['xc2'])[0]
        nequality_idx = np.where(pivot_table['xc1'] != pivot_table['xc2'])[0]

        neq_breakpoints = np.where(np.diff(nequality_idx) >= 8)[0]
        eq_breakpoints = np.where(np.diff(equality_idx) >= 8)[0]

        eq_broken = np.split(equality_idx, eq_breakpoints+1)
        # print(eq_broken)

        neq_broken = np.split(nequality_idx, neq_breakpoints+1)
        # print(neq_broken)

        # print(equality_idx)
        # print(eq_breakpoints)

        plt.figure()
            
        dfs = []
        # df1, df2, df3 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for idx, neq in enumerate(neq_broken):
            try:
                df = pd.concat([pivot_table['xc1'][eq_broken[idx]], pivot_table['xc2'][neq], pivot_table['xc1'][eq_broken[idx+1]]])
            except IndexError:
                pass
            dfs.append(df)
            plt.plot(dfs[idx],'o',label=f'{idx}',markersize=2)
        plt.legend()
        plt.show()
        # plt.figure()

        # for broken in eq_broken:
        #     # print(broken)
        #     plt.plot(pivot_table.iloc[broken]['xc1'],'o',markersize=2, label='xc1')
        #     plt.plot(pivot_table.iloc[broken]['xc2'], 'o', markersize=2, label='xc2')
        #     plt.legend()
        # plt.show()
        # for broken in neq_broken:
        #     # print(broken)
        #     plt.plot(pivot_table.iloc[broken]['xc1'],'o',markersize=2, label='xc1')
        #     plt.plot(pivot_table.iloc[broken]['xc2'], 'o', markersize=2, label='xc2')
        #     plt.legend()
        # plt.show()

        # for i in arrays:
        #     plt.plot(i, 'o', markersize=2)
        # plt.show()

        # for idx, broken_indices in enumerate(neq_broken):
            # print(f"Processing {idx}")
            # for broken in broken_indices:
            #     print(broken)
            #     print(pivot_table.iloc[broken-1:broken+1])