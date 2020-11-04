import json
import time
import random
import string
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from dataset import aug


def plot_json_label(path):
    boxs, onehots = [], []
    with open(path, 'rb') as f:
        data = json.loads(f.read())
        h = data["imageHeight"]
        w = data["imageWidth"]
    canvas = np.zeros((h, w, 3), dtype="uint8") * 255  # 定义画布
    shapes = data["shapes"]
    shapes = aug(shapes)
    for obj in shapes:
        points = obj["points"]
        x = points[0]
        s = points[1]
        t = points[2]
        y = points[3]
        cv2.line(canvas, (int(x[0]), int(x[1])),
                 (int(s[0]), int(s[1])), (255, 0, 0), 2)
        cv2.line(canvas, (int(s[0]), int(s[1])),
                 (int(t[0]), int(t[1])), (255, 0, 0), 2)
        cv2.line(canvas, (int(t[0]), int(t[1])),
                 (int(y[0]), int(y[1])), (255, 0, 0), 2)
        cv2.line(canvas, (int(y[0]), int(y[1])),
                 (int(x[0]), int(x[1])), (255, 0, 0), 2)
    box_path = 'plot/box.jpg'
    cv2.imwrite(box_path, canvas)

    img = cv2.imread(box_path)  # 名称不能有汉字
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)
    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("plot/simfang.ttf", 20,
                              encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    for obj in shapes:
        label = obj["label"]
        points = obj["points"]
        group_id = obj["group_id"]
        y = points[0]
        draw.text((y[0], y[1]), label + ' ' + str(group_id), (255, 255, 0),
                  font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imwrite('plot/show.jpg', cv2charimg)


if __name__ == '__main__':
    import time
    for i in range(100):
        time.sleep(0.5)
        plot_json_label(r'datasets/test/160230555682988.json')
