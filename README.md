# IXMF
## 說明
使用 Autoencoder 和 Random Forest 訓練模型，並透過 Lasso 達成可解釋的極度多標籤學習。

論文：Embedding-based Rule Forests for Interpretable Extreme Multi-label Learning

## 專案架構
- codebase：放置模型與進行實驗會使用到的程式
- experiments：放置實驗內容
- process：放置實驗過程中會產生的模型與圖
- multilabel_datasets：放置多標籤資料集

模型訓練方式請參照 `codebase` 底下的文件

## 需求套件
為了讀取資料集與衡量預測結果，需要額外安裝以下兩個套件。
* [pyxclib](https://github.com/kunaldahiya/pyxclib)
* [scikit-multilearn](http://scikit.ml/)
