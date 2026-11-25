import os
import cv2
import numpy as np
from ultralytics import YOLO

# ex3.py と同じフォルダにある画像を使う
base = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(base, "ex1-26.png")
out_path = os.path.join(base, "ex3_result.png")

# 画像読み込み(日本語を含むパスでも読めるよう imdecode を使用)
img = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 1) YOLOで人物を検出
model = YOLO("yolo11x.pt")
result = model(img, classes=[0], imgsz=1280, conf=0.3, iou=0.8)[0]

# 2) 各人物の頭部領域(枠の上部20%)から青色画素を抽出
#    青: H=100~125, 白っぽく見える遠くのヘルメットも拾えるよう S>=35,
#    紺のユニフォーム(V≒100)を除くため V>=125
LOWER_BLUE = (100, 35, 125)
UPPER_BLUE = (125, 255, 255)
HEAD_RATIO = 0.2   # 人物枠のうち頭部とみなす上部の割合
MIN_PIXELS = 30    # 青ヘルメットと判定する青色画素数の下限

helmets = []
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    hy2 = y1 + int((y2 - y1) * HEAD_RATIO)
    mask = cv2.inRange(hsv[y1:hy2, x1:x2], LOWER_BLUE, UPPER_BLUE)
    if cv2.countNonZero(mask) < MIN_PIXELS:
        continue  # 金ヘルメット(相手チーム)は除外
    # 最大の青色領域をヘルメットとし、その外接矩形を求める
    # (隣の選手のヘルメットと繋がらないよう膨張・クロージングはしない)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    helmets.append((x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh))

# 3) 重なった人物枠で同じヘルメットを二重に数えないよう重複を除去
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0

unique = []
for h in helmets:
    if all(iou(h, u) < 0.3 for u in unique):
        unique.append(h)

# 4) ヘルメット領域を赤枠で描画
PAD = 3
for (x1, y1, x2, y2) in unique:
    cv2.rectangle(img, (x1 - PAD, y1 - PAD), (x2 + PAD, y2 + PAD), (0, 0, 255), 2)

count = len(unique)
print(f"抽出した青いヘルメットの数: {count}")
cv2.putText(img, f"blue helmet: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# 結果の保存と表示
cv2.imencode(".png", img)[1].tofile(out_path)
cv2.imshow("ex3", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
