import cv2
import numpy as np
import os
import json

# パスの設定
face_image_dir = "./data/LFW/archive/generated_yellow-stylegan2"
left_eye_mask_dir = "./data/Dlib_Segmentation_Masks/left_eye"
right_eye_mask_dir = "./data/Dlib_Segmentation_Masks/right_eye"
output_dir = "./data/processed_images_double"
output_json_path = "./eyelid_coordinates.json"

os.makedirs(output_dir, exist_ok=True)

# 二重幅の線を目の端から端まで引く関数
def add_double_eyelid_lines(image, mask, eyelid_offset=-20, line_spacing=5, num_lines=1, eyelid_thickness=2, color=(255, 0, 0)):
    """
    二重幅を目の端から端まで線を引く
    - eyelid_offset: 最初の線を頂点からどれだけ上にオフセットするか
    - line_spacing: 線と線の間隔
    - num_lines: 描画する線の本数
    - eyelid_thickness: 線の太さ
    - color: 線の色 (BGR)
    """
    coordinates = []  # 座標を格納するリスト

    # バイナリマスクを作成
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"検出された輪郭の数: {len(contours)}")  # デバッグ用
    
    for cnt in contours:
        if len(cnt) == 0:
            continue
        
        # 輪郭の最上部を見つける
        top = tuple(cnt[cnt[:, :, 1].argmin()][0])
        print(f"最上部の座標: {top}")  # デバッグ用

        # 輪郭の最左端と最右端を取得
        left = tuple(cnt[cnt[:, :, 0].argmin()][0])
        right = tuple(cnt[cnt[:, :, 0].argmax()][0])
        curve_width = right[0] - left[0]
        print(f"左端: {left}, 右端: {right}, 幅: {curve_width}")  # デバッグ用

        # 複数の曲線を描画
        for line_num in range(num_lines):
            # 各線の基準点（上方向にオフセットして配置）
            curve_top = (top[0], top[1] + eyelid_offset - line_spacing * line_num)

            # 曲線を生成（傾きを緩やかに）
            curve_points = []
            for x_offset in range(-curve_width // 2, curve_width // 2 + 1):
                x = int(curve_top[0] + x_offset)
                y = int(curve_top[1] + 0.01 * (x_offset ** 2)) # 緩やかな二次関数
                curve_points.append((x, y))

            # 曲線を描画
            for i in range(len(curve_points) - 1):
                cv2.line(image, curve_points[i], curve_points[i + 1], color, eyelid_thickness, cv2.LINE_AA)
            
            # 座標を保存
            coordinates.append(curve_points)
    
    return coordinates

# 二重線の座標を保存する関数
def save_eyelid_coordinates(filepath, left_eye_coordinates, right_eye_coordinates):
    data = {
        "left_eye": left_eye_coordinates,
        "right_eye": right_eye_coordinates
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

# メイン処理ループ
left_eye_coordinates = []  # 左目の座標リスト
right_eye_coordinates = []  # 右目の座標リスト

for filename in os.listdir(face_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        face_path = os.path.join(face_image_dir, filename)
        
        # ファイル名から拡張子を除去
        base_name = os.path.splitext(filename)[0]
        
        # マスクファイル名を生成
        left_eye_filename = f"{base_name}_left_eye.png"
        right_eye_filename = f"{base_name}_right_eye.png"
        
        left_eye_path = os.path.join(left_eye_mask_dir, left_eye_filename)
        right_eye_path = os.path.join(right_eye_mask_dir, right_eye_filename)

        # デバッグ用にファイルパスを表示
        print(f"Processing: {filename}")
        print(f"Left eye mask: {left_eye_path}")
        print(f"Right eye mask: {right_eye_path}")

        # 画像とマスクの読み込み
        face_img = cv2.imread(face_path)
        left_eye_mask = cv2.imread(left_eye_path, cv2.IMREAD_GRAYSCALE)
        right_eye_mask = cv2.imread(right_eye_path, cv2.IMREAD_GRAYSCALE)

        if face_img is None:
            print(f"顔画像 {face_path} の読み込みに失敗しました。")
            continue
        if left_eye_mask is None:
            print(f"左目マスク {left_eye_path} の読み込みに失敗しました。")
            continue
        if right_eye_mask is None:
            print(f"右目マスク {right_eye_path} の読み込みに失敗しました。")
            continue

        # 左目に二重幅の線を追加し、座標を収集
        left_eye_coordinates.append(add_double_eyelid_lines(face_img, left_eye_mask))

        # 右目に二重幅の線を追加し、座標を収集
        right_eye_coordinates.append(add_double_eyelid_lines(face_img, right_eye_mask))

        # 結果を保存
        output_path = os.path.join(output_dir, filename)
        success = cv2.imwrite(output_path, face_img)
        if success:
            print(f"処理完了: {output_path}")
        else:
            print(f"結果の保存に失敗しました: {output_path}")

# JSONファイルに座標を保存
save_eyelid_coordinates(output_json_path, left_eye_coordinates, right_eye_coordinates)

print("全ての画像の処理が完了しました。")
