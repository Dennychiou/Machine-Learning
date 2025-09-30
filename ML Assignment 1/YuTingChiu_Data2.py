from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.datasets import load_breast_cancer
import numpy as np

dia = datasets.fetch_openml(data_id=45022)

def BestROCcurve():
  dtc = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1000)
  y_scores = model_selection.cross_val_predict(dtc, dia.data, dia.target, method="predict_proba", cv=10)
  fpr, tpr, th = roc_curve(dia.target, y_scores[:,1],pos_label="1")
  plt.xlabel("1 - Specificity")
  plt.ylabel("Sensitivity")
  plt.xlim(0,1)
  plt.ylim(0,1)
  plt.plot(fpr,tpr,label="Decision Tree")
  plt.legend()
  plt.show()
def AUCvalue_graph():
  X = dia.data
  y = dia.target
  min_samples_leaf_values = [5, 30, 80, 500, 1000, 5000, 10000, 30000, 50000]
  auc_scores = {}
  for min_samples in min_samples_leaf_values:
    dtc = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=min_samples)
    y_scores = cross_val_predict(dtc, X, y, method="predict_proba", cv=10)
    auc_score = roc_auc_score(y, y_scores[:, 1])
    auc_scores[min_samples] = auc_score
  for min_samples, score in auc_scores.items():
    print(f"min_samples_leaf={min_samples}: AUC={score}")
  x_values = list(auc_scores.keys())
  y_values = list(auc_scores.values())
  plt.plot(x_values, y_values, 'o-', color='black') 
  plt.xlabel("min_sample_leaf")
  plt.ylabel("AUC score")
  plt.show()
  
print ("Data2")
(BestROCcurve())
(AUCvalue_graph())
