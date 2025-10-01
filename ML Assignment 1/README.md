# 使用 Scikit-Learn 進行決策樹 AUC 分析
選擇兩個合適的資料(從openML選取)，用決策樹分析調整min_samples_leaf並且做10-fold cross-validation，畫出AUC vs min_samples_leaf，找出最佳參數後畫出ROC curve，最後觀察過擬和與欠擬和的情況。
## 內容
* 針對不同的資料集 ID:1460、ID:45022，分別為A1Data1.py以及A1Data2.py
* 訓練不同值的min_samples_leaf
* 使用10-fold cv評估模型
* AUC圖
* ROC圖
## 資料集條件
* 二元分類任務（Binary classification）
* 所有特徵為數值型（Numeric features）
* 至少包含 1000 筆資料（Instances ≥ 1000）
* 無缺失值（No missing values）
## 步驟以及要求 (可參考A1.txt)
1. 選兩個合適的 OpenML dataset並且符合要符合：
* binary classification
* 全部數值特徵
* ≥1000 筆資料
* 無缺失值
2. 用 Decision Tree (criterion="entropy") 分析
3. 調整 min_samples_leaf 參數，做 10-fold cross-validation
* ⚠️ 注意：至少要測試 5 個以上不同的值，並且涵蓋到「過擬合 (overfitting)」與「欠擬合 (underfitting)」的範圍
4. 畫 AUC vs min_samples_leaf
* ⚠️ 圖上要 標示出 overfitting / underfitting 區域（不是單純畫線就好）
5. 找出最佳參數後，畫 ROC curve
## 執行環境
* Python版本需為3.12以上
* 於 Windows Powershell中安裝指令套件 (pip install scikit-learn openml pandas numpy matplotlib)
