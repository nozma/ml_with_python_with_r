# Pythonではじめる機械学習のRによる学習

- https://nozma.github.io/ml_with_python_with_r/
- [Pythonではじめる機械学習 ―scikit-learnで学ぶ特徴量エンジニアリングと機械学習の基礎](https://www.amazon.co.jp/dp/4873117984/)をRとmlrパッケージを使用してやっていきます。

## Dockerfile

### Usage

`docker run -p 8787:8787 nozma/ml-python-with-r`

### 詳細

`rocker/verse`をベースとした`nozma/ml-python-notebook-r`に以下の変更を加えています。

- Rパッケージの追加
    - GGally
    - mlr
    - mlrCPO
