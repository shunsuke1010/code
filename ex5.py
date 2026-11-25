import os
import cv2
import numpy as np
from ultralytics import YOLO

# ex5.py と同じフォルダにある動画・モデルを使う
base = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(base, "ex5-26.mp4")
out_path = os.path.join(base, "ex5_result.mp4")
model_path = os.path.join(base, "best260408.pt")

# 青いヘルメット(KG_helmet)検出用の学習済みモデル
# クラスは {0: 'KG_helmet'} の1クラスのみ
model = YOLO(model_path)

# 動画の読み込みと、結果を書き出す動画の準備(元動画と同じfps・サイズ)
cap = cv2.VideoCapture(in_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1フレームごとにヘルメットを検出(入力サイズは動画の幅1280に合わせる)
    result = model(frame, imgsz=1280, conf=0.25, verbose=False)[0]

    boxes = [list(map(int, box.xyxy[0])) for box in result.boxes]

    # 誤検出の除去: ユニフォームの一部などを小さな枠で誤検出することがあるため、
    # そのフレームの枠面積の中央値の40%未満の小さい枠は除外する
    if boxes:
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        min_area = 0.4 * np.median(areas)
        boxes = [b for b, a in zip(boxes, areas) if a >= min_area]

    # 検出したヘルメットを赤枠で描画
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGRで赤
    count = len(boxes)
    cv2.putText(frame, f"blue helmet: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 30フレームごとに検出したヘルメット数をターミナルに表示
    if frame_no % 30 == 0:
        print(f"frame {frame_no}: 検出したヘルメットの数 = {count}")

    # 結果の保存と表示(qキーで途中終了)
    writer.write(frame)
    cv2.imshow("ex5", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_no += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
