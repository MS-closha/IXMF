# XML
使用 Autoencoder 和 Random Forest 達成可解釋的極度多標籤學習。

## 需求套件
為了讀取資料集與衡量預測結果，需要額外安裝以下兩個套件。
* [pyxclib](https://github.com/kunaldahiya/pyxclib)
* [scikit-multilearn](http://scikit.ml/)

## 資料集
在 `dataset.py` 內定義了3個資料集的讀取方式，需要下載資料集並將資料夾 `multilabel_datasets` 存放至指定路徑，或依照需要直接修改檔案路徑。

## 使用方式
### 訓練 Autoencoder
Autoencoder 的模型架構皆定義在 `model.py` 中。

可以在終端機中輸入以下指令用預設參數來執行 `train_AE.py`。
```
python3 train_AE.py
```
可以使用的參數包含 `--dataname`、 `--epochs` 和 `--lr`，分別指定訓練用的資料集、訓練次數和 learning rate。
例如 
```
python3 train_AE.py --dataname 'bibtex' --epochs 20 --lr 0.0001 
```
訓練完的結果會儲存在 `AE/` 資料夾中。
### 訓練隨機森林模型
訓練與預測的主要函式定義於 `utils.py` 中。

在終端機中輸入以下指令執行 `train.py` 可以直接使用已訓練好的 Autoencoder 模型來編碼資料，並訓練隨機森林，訓練完的模型會自動儲存在 `models/` 目錄中，訓練表現和參數會以 `.csv` 的格式儲存在 `outcome/` 資料夾中。

```
python3 train.py 
```
可以使用的參數包含 `--dataname`、 `--tree-per-label`、 `--max_features`、 `--max_depth` 和 `--n_jobs`，分別指定訓練用的資料集和隨機森林的相關參數。
```
python3 train.py -d 'bibtex' -t 3 -f 0.8 -p 15 -j 8
```
`'bibtex'`資料在 `tree-per-label=3`、 `--max_features＝0.8`、 `max_depth=None` 、`n_jobs=1` 的情況下約需要訓練 5分鐘。

### 其他
`evaluation.py` 存放讀取模型並評估表現的程式碼，可以根據需求修改使用。