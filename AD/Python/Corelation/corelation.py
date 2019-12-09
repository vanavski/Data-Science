import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("ONLINEADS.xlsx",sheet_name = 0)

#corelation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()