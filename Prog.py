import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
import sklearn

csv = np.genfromtxt ('target.csv', delimiter=",")
target = csv[:,0]

y=[int(i)for i in target]
Y=np.array(y)
import csv
with open('data.csv', 'rb') as f:
    reader = csv.reader(f)
    lst = list(reader)
    x = [[(float(j)) for j in i] for i in lst]
    x = [np.array(i) for i in x]
X=np.array(x)



clf = GaussianNB()
clf.fit(X,Y)
print clf.predict([X[30]])