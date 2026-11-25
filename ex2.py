import os
import cv2
import numpy as np
from ultralytics import YOLO

# ex2.py と同じフォルダにある画像を使う
base = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(base, "ex2-26.png")
out_path = os.path.join(base, "ex2_result.png")

# 画像読み込み(日本語を含むパスでも読めるよう imdecode を使用)
img = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), cv2.IMREAD_COLOR)

# 学習済みYOLOモデル(初回実行時に自動ダウンロード)
model = YOLO("yolo11x.pt")

# person(COCOのクラスID 0)のみ検出
# 密集して重なった選手も拾えるよう信頼度閾値を下げ(0.15)、
# 同じ人物への二重検出を消すためNMSのIoU閾値は0.5にする
result = model(img, classes=[0], imgsz=1280, conf=0.15, iou=0.5)[0]

# 中心人物(フィールド上の選手・審判)のみ残す
# 足元(枠の下端)のy座標が、奥のサイドライン(y≈250)より手前、
# 手前のベンチ・スタッフ(y≈600以降)より奥にある人物を採用
# → 観客席・サイドラインの控え選手・手前のスタッフを除外
FIELD_TOP, FIELD_BOTTOM = 250, 560

count = 0
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    if not (FIELD_TOP < y2 < FIELD_BOTTOM):
        continue
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGRで赤
    count += 1

# 人物の数を表示
print(f"検出した人物の数: {count}")
cv2.putText(img, f"person: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# 結果の保存と表示
cv2.imencode(".png", img)[1].tofile(out_path)
cv2.imshow("ex2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
