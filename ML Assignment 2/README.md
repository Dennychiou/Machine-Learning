# Scikit-Leaarn多種模型分類比較
從 OpenML中選取2個資料集，並測試五種不同的模型，使用cross_val_score 報告 10-fold CV 的AUC平均值與標準差
## 內容
* DecisionTreeClassifier (sklearn.tree):
調整參數：min_samples_leaf（至少 5 個不同數值）。
* KNeighborsClassifier (sklearn.neighbors):
調整參數：n_neighbors（至少 5 個不同數值）。
* MultinomialNB (sklearn.naive_bayes):
使用預設參數即可。
* LogisticRegression (sklearn.linear_model):
使用預設參數即可。
* DummyClassifier (sklearn.dummy):
基準模型，只輸出最常見的類別。
## 資料集條件
* 分類任務（target 為 nominal，二元或多元都可以）。
* 至少 一個類別型（categorical/nominal）特徵。
* 至少 1000 筆資料。
## 步驟以及要求 (可參考A2.txt)
