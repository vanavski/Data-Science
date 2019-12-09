import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("creditcard.csv")

#corelation
plt.figure(figsize=(22,18))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()