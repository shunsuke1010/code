import os
import cv2
import numpy as np
from ultralytics import YOLO

# ex4.py と同じフォルダにある画像・モデルを使う
base = os.path.dirname(os.path.abspath(__file__))
in_path = os.path.join(base, "ex2-26.png")
out_path = os.path.join(base, "ex4_result.png")
model_path = os.path.join(base, "best260408.pt")

# 画像読み込み(日本語を含むパスでも読めるよう imdecode を使用)
img = cv2.imdecode(np.fromfile(in_path, dtype=np.uint8), cv2.IMREAD_COLOR)

# 青いヘルメット(KG_helmet)検出用の学習済みモデル
# クラスは {0: 'KG_helmet'} の1クラスのみ
model = YOLO(model_path)

# ヘルメットは10px程度と小さいため入力サイズを1280に拡大して検出
result = model(img, imgsz=1280, conf=0.25)[0]

# 検出したヘルメットを赤枠で描画
count = 0
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # BGRで赤
    count += 1

# ヘルメットの数を表示
print(f"抽出した青いヘルメットの数: {count}")
cv2.putText(img, f"blue helmet: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

# 結果の保存と表示
cv2.imencode(".png", img)[1].tofile(out_path)
cv2.imshow("ex4", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
