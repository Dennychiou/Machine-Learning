# 情緒圖片分類任務 — 自建 CNN 與遷移學習模型比較
本專案目標是使用 Keras 深度學習框架，對六種人類情緒圖片進行分類，並比較不同 CNN 模型架構與預訓練模型（VGG16）的效能。
## 🧠專案目標
1. Task 1：建立兩個 CNN 模型並比較
2. Task 2：使用預訓練模型 EfficientNetV2B3
3. Task 3：實際圖片情緒預測與錯誤分析
# ⚙️資料前處理流程
## Task 1：自建 CNN 模型 📐 模型架構
輸入：100×100 RGB 圖片
模型流程：
1. Conv2D (32 filters) → 抓取基本特徵
2. MaxPooling → 降維
3. Conv2D (64 filters) → 抓更深層特徵
4. MaxPooling → 降維
5. Dropout (0.2, 0.3) → 防止過擬合
6. Flatten → 攤平成向量
7. Dense(128, ReLU) → 學習情緒特徵
8. Dropout(0.5)
9. Softmax Output → 多分類情緒預測
* 用卷積層抓臉部特徵，再用全連接層做分類。
* Model 1 和 Model 2的比較為解析度更高 → 可以看到更細微的表情細節，但代價是：運算更慢、容易過擬合
## Task 2：使用預訓練模型
* 新增層：GlobalAveragePooling
* 📊 Task2 結果

| 模型             | Test Accuracy |
| ---------------- | ------------- |
| Task1 最佳模型    | 33.0%         |
| EfficientNetV2B3 | 69.7%         |
## Task 3: 錯誤分析
測試自選的 10 張情緒圖片
比較 Task 1 與 Fine-tuned 模型的分類結果
產出圖片視覺化與預測標籤差異分析報告

