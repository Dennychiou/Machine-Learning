from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn import datasets
import numpy as np

bm = datasets.fetch_openml(data_id=1461)
X, y = bm.data, bm.target

def DTC():
    nominal_features = X.select_dtypes(include=["category", "object"]).columns
    numeric_features = X.select_dtypes(exclude=["category", "object"]).columns

    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), nominal_features)
        ],
        remainder='passthrough'
    )

    X_encoded = ct.fit_transform(X)

    dtc = DecisionTreeClassifier()
    parameters = {'min_samples_leaf': [5, 20, 50, 200, 500]}

    tuned_dtc = GridSearchCV(dtc, parameters, scoring="roc_auc", cv=10)
    cv_result = cross_val_score(tuned_dtc, X_encoded, y, cv=10, scoring="roc_auc")
    print("DTC - Best Mean AUC:", cv_result.mean())
    print("DTC - Standard Deviation of AUC:", cv_result.std())
    print("-" * 50)

def KNN():
    nominal_features = X.select_dtypes(include=["category", "object"]).columns
    numeric_features = X.select_dtypes(exclude=["category", "object"]).columns

    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), nominal_features)
        ],
        remainder='passthrough'
    )

    X_encoded = ct.fit_transform(X)

    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [5, 20, 50, 80, 100]}

    tuned_knn = GridSearchCV(knn, parameters, scoring="roc_auc", cv=10)
    cv_result = cross_val_score(tuned_knn, X_encoded, y, cv=10, scoring="roc_auc")

    print("KNN - Best Mean AUC:", cv_result.mean())
    print("KNN - Standard Deviation of AUC:", cv_result.std())
    print("-" * 50)

def MNB():
    nominal_features = X.select_dtypes(include=["category", "object"]).columns
    numeric_features = X.select_dtypes(exclude=["category", "object"]).columns

    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), nominal_features),
            ('num', MinMaxScaler(), numeric_features)
        ],
        remainder='passthrough'
    )

    X_encoded = ct.fit_transform(X)

    mnb = MultinomialNB()
    
    cv_result = cross_val_score(mnb, X_encoded, y, cv=10, scoring="roc_auc")

    print("MNB - Best Mean AUC:", cv_result.mean())
    print("MNB - Standard Deviation of AUC:", cv_result.std())
    print("-" * 50)

def LR():
    nominal_features = X.select_dtypes(include=["category", "object"]).columns
    numeric_features = X.select_dtypes(exclude=["category", "object"]).columns

    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), nominal_features)
        ],
        remainder='passthrough'
    )

    X_encoded = ct.fit_transform(X)

    lr = LogisticRegression(solver='liblinear')

    cv_result = cross_val_score(lr, X_encoded, y, cv=10, scoring="roc_auc")
    print("LR - Best Mean AUC:", cv_result.mean())
    print("LR - Standard Deviation of AUC:", cv_result.std())
    print("-" * 50)

def DC():
    nominal_features = X.select_dtypes(include=["category", "object"]).columns
    numeric_features = X.select_dtypes(exclude=["category", "object"]).columns

    ct = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False), nominal_features)
        ],
        remainder='passthrough'
    )

    X_encoded = ct.fit_transform(X)
    
    dummy_clf = DummyClassifier(strategy='most_frequent')
    
    cv_result = cross_val_score(dummy_clf, X_encoded, y, cv=10, scoring="roc_auc")
    print("DC - Best Mean AUC:", cv_result.mean())
    print("DC - Standard Deviation of AUC:", cv_result.std())
    print("-" * 50)

print("DTC value")
(DTC())
print("KNN value")
(KNN())
print("MNB value")
(MNB())
print("LR value")
(LR())
print("DC value")
(DC())
