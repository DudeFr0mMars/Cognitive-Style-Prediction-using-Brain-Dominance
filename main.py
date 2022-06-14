import pandas
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score

df = pandas.read_csv("Psychology Project WINSEM 2021-2022 - Responses.csv")
df.rename(columns={'205 Approved out of 250':'Status'},inplace=True)

d = {'Approved': 1, 'Rejected':0}
df['Status'] = df['Status'].map(d)
d = {'Undifferentiated Style': 1, 'Split Style': 2,'Intuitive Style': 3,'Systematic Style': 4,'Integrated Style': 5}
df['Cognitive Style'] = df['Cognitive Style'].map(d)
d = {'No clear preference': 1, 'Left Brained':2,'Right Brained':3}
df['Dominance'] = df['Dominance'].map(d)
d = {'18-30': 2, '<18': 1,'31-50':3,'>50': 4}
df['Age '] = df['Age '].map(d)

df = df[df['Status']==1]
train,test = train_test_split(df,test_size = 0.2)
features = ['Age ','A','B','Total Systematic Score','Total Intuitive Score']
X = train[features]
y = train['Cognitive Style']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)
graph.write_png('mydecisiontree.png')

img=pltimg.imread('mydecisiontree.png')
imgplot = plt.imshow(img)
plt.show()

predictions = dtree.predict(test[features])
cmat = confusion_matrix(test['Cognitive Style'],predictions) 
print(cmat)
 
