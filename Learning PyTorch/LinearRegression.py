import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
%matplotlib inline

sns.set_sytle(style = 'whitegrid')
plt.rcParams["patch.force_edgecolor"] = True

m = 2
c = 3
c = 3
x = np.random.rand(256)
noise = np.random.randn(256)/4
y = x * m + c + noise

df = pd.Dataframe()
df['x'] = x
df['y'] = y
sns.lmplot(x = 'x', y = 'y', data = df)