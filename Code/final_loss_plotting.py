# prompt: load csv data from 4 files with name in the form "loss-mode-6-width-32.csv" for mode is 6, 8, and 12 and width is 16, 32, and 64

import os
import pandas as pd
import matplotlib.pyplot as plt


modes = [6]
widths = [32, 64, 128]

dfs = {}

for loss_mode in modes:
  for width in widths:
    filename = f"losses-BIG-mode-{loss_mode}-width-{width}.csv"
    filepath = os.path.join('/groups/mlprojects/eikonal/FinalizedLosses', filename)
    try:
        df = pd.read_csv(filepath)
        df['Log Test Smooth'] = df['Log Test'].rolling(window=20).mean()
        df['Log Train Smooth'] = df['Log Train'].rolling(window=20).mean()
        dfs[loss_mode, width] = df
    except:
        continue
    
    
# loss plots smooth

for mode, width in dfs:
    plt.plot(dfs[mode, width]["Log Train Smooth"])

plt.legend([f"Mode {mode}, Width {width}" for mode, width in dfs])
plt.title("Train Loss Smooth")
plt.show()

for mode, width in dfs:
    plt.plot(dfs[mode, width]["Log Test Smooth"])

plt.legend([f"Mode {mode}, Width {width}" for mode, width in dfs])
plt.title("Test Loss Smooth")
plt.show()


# loss plots smooth


for mode, width in dfs:
    plt.plot(dfs[mode, width]["Log Train"])

plt.legend([f"Mode {mode}, Width {width}" for mode, width in dfs])
plt.title("Train Loss")
plt.show()

for mode, width in dfs:
    plt.plot(dfs[mode, width]["Log Test"])

plt.legend([f"Mode {mode}, Width {width}" for mode, width in dfs])
plt.title("Test Loss")
plt.show()