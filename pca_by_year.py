#%%
import matplotlib.ticker as ticker
from pandas import plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn  # 機械学習のライブラリ
from sklearn.decomposition import PCA  # 主成分分析器

df = pd.read_csv('./data_by_year.csv')
plotting.scatter_matrix(df.iloc[:, 1:], figsize=(
    8, 8), c=list(df.iloc[:, 0]), alpha=0.5)
plt.show()
#%%
df = df.set_index('year')
# 行列の標準化
dfs = df.iloc[:, 1:-1].apply(lambda x: (x-x.mean())/x.std(), axis=0)
#主成分分析の実行
pca = PCA()
pca.fit(dfs)
# データを主成分空間に写像
feature = pca.transform(dfs)
#%%
# 主成分得点
print(pd.DataFrame(feature, columns=["PC{}".format(x + 1)
                               for x in range(len(dfs.columns))]).head())
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(18, 18))
for x, y, name in zip(feature[:, 0], feature[:, 1], df.index):
    plt.text(x, y, name)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=df.index)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %%
# 寄与率
print(pd.DataFrame(pca.explained_variance_ratio_, index=[
    "PC{}".format(x + 1) for x in range(len(dfs.columns))]))
# 累積寄与率を図示する
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()

# %%
# PCA の固有値
print(pd.DataFrame(pca.explained_variance_, index=[
             "PC{}".format(x + 1) for x in range(len(dfs.columns))]))

# %%
# PCA の固有ベクトル
print(pd.DataFrame(pca.components_, columns=df.columns[1:-1], index=[
             "PC{}".format(x + 1) for x in range(len(dfs.columns))]))

# %%
# 第一主成分と第二主成分における観測変数の寄与度をプロットする
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[1:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %%
