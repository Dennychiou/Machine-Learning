# 使用 TensorFlow/Keras 建立簡單前饋神經網路（Feedforward Neural Network）
本專案對 OpenML 上的兩個回歸資料集進行訓練與評估，並比較不同隱藏層神經元數量對模型效能（MSE）的影響。
## 🧠專案目標
* 使用神經網路解決回歸問題
* 比較不同模型大小（units）對預測能力的影響
* 觀察訓練與驗證誤差曲線
* 評估測試集 MSE
* 以表格形式輸出結果
1. Dataset 1：(data_id=287)
2. Dataset 2：(data_id=503)
# ⚙️資料前處理流程
## 處理步驟：
* 將目標值轉為 2D array
* 分割資料：
1. 訓練集 80%
2. 測試集 20%
* 使用 StandardScaler 標準化：
1. 特徵 (X)
2. 目標值 (y)
* 再從訓練集中切出驗證集：
1. 訓練集 80%
2. 驗證集 20%
## 📈訓練設定
* Loss：MSE (Mean Squared Error)
* Optimizer：Adam
* Epochs：50
* 評估指標：MSE
對每組資料集皆使用以下三種神經網路架構：

1. Model 1：少量神經元
隱藏層神經元數量：10
2. Model 2：適量神經元
隱藏層神經元數量：100
3. Model 3：過多神經元
隱藏層神經元數量：400

| Model   | Hidden Units |
| ------- | ------------ |
| Model 1 | 10           |
| Model 2 | 100          |
| Model 3 | 400          |
# 📊 輸出結果
1.顯示每個模型的訓練曲線
2.評估測試集 MSE
3.以表格顯示結果

| Dataset  |  Model  |  Test MSE|
| ---------------| ------- |------- |
| 0  Dataset 287 | Model 1 | 0.1234 |
| 1  Dataset 287 | Model 2 | 0.0987 |
| 2  Dataset 287 | Model 3 | 0.0856 |
| 3  Dataset 503 | Model 1 | 0.4567 |
| 4  Dataset 503 | Model 2 | 0.3890 |
| 5  Dataset 503 | Model 3 | 0.3721 |
