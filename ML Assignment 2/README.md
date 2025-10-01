# Scikit-Leaarn多種模型分類比較
從 OpenML中選取2個資料集，並測試五種不同的模型，使用cross_val_score 報告 10-fold CV 的AUC平均值與標準差
## 內容
### 針對不同的資料集 ID:1460、ID:45022，分別為A2Data1.py以及A2Data2.py
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
1. 選兩個合適的 OpenML dataset並且符合要符合：
* Target → Nominal（分類）。
* Instances ≥ 1000。
* 至少一個 Nominal feature。
* 避免 Instances > 100,000。
* ⚠️ 注意: 當處理類別特徵(如target 以外的 features 有字串/類別型資料) 使用One-Hot Encoding。
2. 使用 GridSearchCV 調整參數
3. 用 cross_val_score 計算平均與標準差
## 結果呈現
## 📊 Cross-Validation Results

| Model                | Best Mean AUC | Std. AUC |
|-----------------------|---------------|----------|
| Decision Tree (DTC)  | 0.8529        | 0.0231   |
| K-Nearest Neighbors (KNN) | 0.8622   | 0.0165   |
| Multinomial Naive Bayes (MNB) | 0.8057 | 0.0394   |
| Logistic Regression (LR) | 0.8640   | 0.0296   |
| Dummy Classifier (DC) | 0.5000        | 0.0000   |

## 執行環境
* Python版本需為3.12以上
* 於 Windows Powershell中安裝指令套件 (pip install scikit-learn openml pandas numpy matplotlib)
