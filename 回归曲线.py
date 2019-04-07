import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression

pata='C:/Users/asus/Desktop/'
df=pandas.read_csv(pata+'iris.data.csv', names=['sepal length','sepal width','petal length','petal width','class'])
model=LinearRegression()
a=df['sepal width'][:50]
b=df['sepal length'][:50]
x=np.array(a).reshape(-1,1)
y=np.array(b).reshape(-1,1)
model.fit(x,y)
X=np.arange(2.0,6.0,0.1).reshape(-1,1)
Y=model.predict(X)
fig,ax=plt.subplots(figsize=(7,7))
ax.plot(Y,X,'blue')
ax.scatter(b,a,label='data oeint',color='g' )
ax.set_ylabel('Spepal Length')
ax.set_xlabel('Spepal width')
ax.set_title('Setosa  Sepal Width VS.sepl Length',fontsize=14,y=1.02)

plt.show()
os.system("pause")
