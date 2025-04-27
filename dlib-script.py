import os
import cv2
import numpy as np
import dlib
from tqdm import tqdm

# 入力ディレクトリと出力ディレクトリの設定
input_dir = "./data/LFW/archive/generated_yellow-stylegan2"  # StyleGAN2生成画像ディレクトリ
output_dir = "./data/Dlib_Segmentation_Masks"

# 出力ディレクトリ内の各パーツ用サブディレクトリを作成
parts = ['left_eye', 'right_eye', 'nose', 'mouth', 'left_eyebrow', 'right_eyebrow', 'jaw']
for part in parts:
    os.makedirs(os.path.join(output_dir, part), exist_ok=True)

# Dlibのモデルファイル
predictor_path = "./data/Dlib/shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    print("事前学習モデルが見つかりません。以下からダウンロードしてください：")
    print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

# Dlibの顔検出器とランドマーク予測器を初期化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 顔パーツのランドマークインデックス
face_parts_landmarks = {
    'jaw': list(range(0, 17)),
    'left_eyebrow': list(range(17, 22)),
    'right_eyebrow': list(range(22, 27)),
    'nose': list(range(27, 36)),
    'left_eye': list(range(36, 42)),
    'right_eye': list(range(42, 48)),
    'mouth': list(range(48, 68)),
}

# カラーマップの定義（各パーツに異なる色を割り当て）
part_colors = {
    'left_eye': (255, 0, 0),        # 赤
    'right_eye': (0, 255, 0),       # 緑
    'nose': (0, 0, 255),            # 青
    'mouth': (255, 255, 0),         # シアン
    'left_eyebrow': (255, 0, 255),  # マゼンタ
    'right_eyebrow': (0, 255, 255), # 黄
    'jaw': (128, 128, 128)          # グレー
}

# マスク生成関数
def generate_masks(image, landmarks):
    height, width, _ = image.shape
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    for part, indices in face_parts_landmarks.items():
        points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in indices], dtype=np.int32)
        # 特定のパーツに閉じたポリゴンを描画
        cv2.fillPoly(mask, [points], part_colors[part])

    return mask

# 各パーツごとのマスクを保存する関数
def save_part_masks(mask, image_filename):
    for part in parts:
        part_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        color = part_colors[part]
        # 色でフィルタリングしてマスクを生成
        condition = np.all(mask == color, axis=2)
        part_mask[condition] = 255
        # ファイル名にパーツ名を追加
        base_filename = os.path.splitext(image_filename)[0]
        mask_filename = f"{base_filename}_{part}.png"
        mask_path = os.path.join(output_dir, part, mask_filename)
        cv2.imwrite(mask_path, part_mask)

# 画像の処理
def process_images(input_directory):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(valid_extensions)]

    print(f"Processing {len(image_files)} images in {input_directory}...")

    for image_file in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_directory, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"画像の読み込みに失敗しました: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            print(f"顔が検出されませんでした: {image_path}")
            continue

        # 最初の顔のみ処理
        landmarks = predictor(gray, faces[0])

        # マスクの生成
        mask = generate_masks(image, landmarks)

        # 各パーツごとにマスクを保存
        save_part_masks(mask, image_file)

# メイン処理
def main():
    process_images(input_dir)
    print("セグメンテーションマスクの生成が完了しました。")

if __name__ == "__main__":
    main()
