# Python_Face

## 概要

本リポジトリには、以下の 2 つのスクリプトが含まれます。

- `dlib-script.py`  
  　 Dlib を用いて顔パーツのセグメンテーションマスクを生成するスクリプト
- `eyes-output.py`  
  　生成されたマスクを使用して、目に二重線を描画するスクリプト

## 必要環境

本プロジェクトの実行には、以下の環境が必要です。

- Python 3.7 以上

インストールされていない場合は、[公式サイト](https://www.python.org/)よりインストールしてください。

## 必要な Python ライブラリ

以下のコマンドで必要なパッケージをインストールしてください。

```bash
pip install numpy opencv-python dlib tqdm
```

インストールに失敗する場合は、pip3 を使用してください。

```
pip3 install numpy opencv-python dlib tqdm
```

## 事前準備

### Dlib モデルファイルのダウンロード

`dlib-script.py`は、顔のランドマーク検出に Dlib の学習済みモデルを使用します。以下の手順でファイルを準備してください。

1. 以下の URL より`generated_yellow-stylegan2.dat.bz2`をダウンロード
   [aisanfaces](https://www.kaggle.com/datasets/lukexng/aisanfaces)
2. ダウンロードしたファイルを解凍
3. 解凍後の`generated_yellow-stylegan2.dat`ファイルを、以下のパスに配置します。

```
Python_Face/data/Dlib/
```

※フォルダが存在しない場合は手動で作成してください。

## ディレクトリ構成

実行前に、最低限以下のフォルダ構成を作成しておく必要があります。

```
Python_Face/
├── dlib-script.py
├── eyes-output.py
├── data/
│   ├── Dlib/
│   │   └── shape_predictor_68_face_landmarks.dat
│   └── LFW/
│       └── archive/
│           └── generated_yellow-stylegan2/
│               ├── (顔画像ファイル)
│               ├── (顔画像ファイル)
```

＊`generated_yellow-stylegan2/`配下には処理対象の顔画像（JPEG、PNG 形式）を格納してください。

## 実行手順

1. 顔パーツマスクの生成
   ターミナルで以下のコマンドを実行してください。

```
cd Python_Face
python dlib-script.py
```

処理が完了すると、以下のディレクトリに顔パーツごとのマスク画像が出力されます。

```
Python_Face/data/Dlib_Segmentation_Masks/
```

2. 二重線描画処理の実行
   続けて、以下のコマンドを実行してください。

```
python eyes-output.py
```

処理が完了すると、二重線が描画された顔画像が以下に保存されます。

```
Python_Face/data/processed_images_double/
```

また、二重線の座標情報が以下に JSON ファイルとして出力されます。

```
Python_Face/eyelid_coordinates.json
```

### 注意事項

- 画像に顔が検出されなかった場合、対象ファイルはスキップされます。
- 左右の目のマスクが存在しない場合も対象ファイルはスキップされます。
- 保存時にエラーが発生した場合は、パスやファイル名を確認してください。
- 本スクリプトは一部処理中に標準出力（ターミナル）にログを出力します。デバッグ用途としてご確認ください。
