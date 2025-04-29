# House Prices（Pytorch NN）

本プロジェクトは、[KaggleのHousePricesコンペ]([https://www.kaggle.com/c/titanic](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview))で住宅の特徴から価格を予測するモデルを作成します。

### 使用技術など
- Python 3.10.9
- Numpy
- Pandas
- Seaborn
- matplotlib
- Scikit-Learn
- Pytorch

## 目次
1. HousePricesコンペ
2. 特徴量エンジニアリング
3. モデル定義
4. チューニング
5. 結果

## 1.概要
住宅の特徴から価格を予測するということで問題は回帰問題となります。そのため、回帰モデルとして様々なモデルから多様なアプローチできるため、前回のTitanicコンペと比べると難易度が高いと思います。
今回はディープラーニングの実装もかねて、Pytorchを用いてニューラルネットワークを使った回帰モデルを構築しました。

## 2.特徴量エンジニアリング
行った特徴量エンジニアリングをまとめます。
- 不要な特徴量を削除
- 目的変数を対数変換する
- 特徴量をデータ型別に分けて前処理。(数値列：欠損値→中央値→標準化),(カテゴリ列:欠損値→"Missing"→One-Hot Encoding)
## 3.モデル定義
Pytorchを使ったNNモデルを作成しました。モデルの可視化方法がわからないため図を用意できませんが、Linear→ReLU→BatchNorm1d←Dropoutを繰り返した単純なモデルとなります。

## 4.チューニング
epoch数は60としています。理由として10~100で学習を試した結果、60より増やしてもMSElossの改善が見込めなかったためです。

## 5.結果
結果として提出スコアは0.35244で、4215/24858位(2025/4/29時点)でした。
今回のモデルを実装する前、ChatGPTを用いて簡単なNNモデルを実装し予測させた結果、0.40859だったためスコアとしては改善されたと言えます。
