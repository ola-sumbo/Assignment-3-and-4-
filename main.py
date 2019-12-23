import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import seaborn as sns

url = 'https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv'
df = pd.read_csv(url, sep = ',')
print(df.columns)
df = DataFrame(df)
print(df.head())
print(df.describe())

# pivot table creation
table = df.pivot_table(index=['country'], columns=['year'], values=['lifeExp'])
table.plot()
print(table)

# heat map
# sns.heatmap(table).get_figure().savefig('heatmap.png')

#define the plot
#fig, ax = plt.subplots(figsize=(12,7))

#add title to heat map
#title = "Life-expectancy by Countries over the Years"

# Set the distance and the font size of the title from the plot
#plt.title(title, fontsize=15)
#ttl = ax.title
#ttl.set_position([0.5,1.0])

#hide ticks for x and y axis
#ax.set_xticks([])
#ax.set_yticks([])

#remove the axis
#ax.axis('off')

#sns.heatmap(table, annot=True,fmt="",cmap='RdYlGn',linewidths=0.30,ax=ax)
#plt.show()




# alternatively
hmp1 = df[['country','year','lifeExp']]

# pivot with various variables
hmp2 = pd.pivot_table(hmp1,values='lifeExp',index=['country'],columns='year')
print(hmp2.head(20))
plt.figure(figsize=(8,12))
sns.heatmap(hmp2, cmap="RdBu").get_figure().savefig('hmp2.png')
