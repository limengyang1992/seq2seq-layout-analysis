import json
import time
import random
import string
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import anyconfig

config = anyconfig.load(open("config.yaml", 'rb'))
model = Doc2Vec.load('doc2vec/model')

keys = list(config["label"]["id2value"].keys())[1:]

dic_label = {}
for i,key in enumerate(keys):
    dic_label[key] = i+1
print(dic_label)


def aug(shapes):
    add_shapes = []
    rm_shapes = []
    for shape in shapes:
        points = shape['points']
        points = sorted(points)
        if points[0][1] > points[1][1] and points[2][1] > points[3][1]:
            points = [points[1], points[3], points[2], points[0]]
        elif points[0][1] > points[1][1] and points[2][1] < points[3][1]:
            points = [points[1], points[2], points[3], points[0]]
        elif points[0][1] < points[1][1] and points[2][1] > points[3][1]:
            points = [points[0], points[3], points[2], points[1]]
        else:
            points = [points[0], points[2], points[3], points[1]]
        label = shape['label']
        group_id = shape['group_id']
        types = random.randint(0, 15)
        # types = 4
        # 切分随机加
        if types == 3:
            shape1 = {}
            shape2 = {}
            if len(label) < 2:
                continue
            nums = (points[1][0] - points[0][0]) / 2
            # 左侧
            label1 = label[:len(label) // 2]
            points1 = [[points[0][0], points[0][1]], [points[0][0] + nums, points[1][1]],
                       [points[3][0] + nums, points[2][1]], [points[3][0], points[3][1]]]
            shape1['label'] = label1
            shape1['points'] = deviation(points1)
            shape1['group_id'] = shape['group_id']
            shape1['shape_type'] = shape['shape_type']
            shape1['flags'] = shape['flags']
            add_shapes.append(shape1)
            # 右侧
            label2 = label[len(label) // 2:]
            # points2[0][0] = points2[0][0] + nums
            # points2[3][0] = points2[3][0] + nums
            points2 = [[points[0][0] + nums, points[0][1]], [points[1][0], points[1][1]],
                       [points[2][0], points[2][1]], [points[3][0] + nums, points[3][1]]]
            shape2['label'] = label2
            shape2['points'] = deviation(points2)
            shape2['group_id'] = shape['group_id']
            shape2['shape_type'] = shape['shape_type']
            shape2['flags'] = shape['flags']
            add_shapes.append(shape2)
            rm_shapes.append(shape)
            # shapes.remove(shape)
        # 随机换字
        elif types == 4:
            # 数字
            if group_id in (91, 101, 121):
                word = ''
                for i in range(len(label)):
                    temp = random.randint(0, 9)
                    word = word + str(temp)
                label = word
            # 金钱
            elif group_id in (141, 151, 171):
                l = random.sample(range(0, 9), random.randint(3, 9))
                l.insert(-2, '.')
                label = '￥' + ' ' + ''.join('%s' % id for id in l)  # 数字字母
            elif group_id in (41, 21):
                population = string.ascii_uppercase + string.digits * 5
                letterlist = random.sample(population, len(label))
                label = ''.join(letterlist)
            # 时间
            elif group_id == 111:
                # 设置开始日期时间元组（1976-01-01 00：00：00）
                a1 = (1976, 1, 1, 0, 0, 0, 0, 0, 0)
                # 设置结束日期时间元组（2020-12-31 23：59：59）
                a2 = (2020, 12, 31, 23, 59, 59, 0, 0, 0)
                start = time.mktime(a1)  # 生成开始时间戳
                end = time.mktime(a2)  # 生成结束时间戳
                t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
                date_touple = time.localtime(t)  # 将时间戳生成时间元组
                # 将时间元组转成格式化字符串（1976-05-21）
                date = time.strftime("%Y-%m-%d", date_touple)
                # word = date
                date = date.split('-')
                label = date[0] + "年" + date[1] + "月" + date[2] + "日"
            # 百分数
            elif group_id == 161:
                temp = random.randint(0, 20)
                label = str(temp) + '%'
            else:
                label = label
            shape['label'] = label
        else:
            points = deviation(points)
            shape['points'] = points
        for point in points:
            if point[0] < 0:
                point[0] = 0
            if point[1] < 0:
                point[1] = 0
    for rm_shape in rm_shapes:
        shapes.remove(rm_shape)
    remove_list = random.sample(range(0, len(shapes)), random.randint(2, 5))
    remove_list = sorted(remove_list, reverse=True)
    # for i in remove_list:
    #     shapes.pop(i)
    shapes = shapes + add_shapes
    return shapes


# 偏移量
def deviation(points):
    offset = random.randint(-5, 5)
    indexs = random.randint(1, 4)
    n = random.randint(0, 2)
    if indexs == 4:
        # 上下左右都加
        if n == 0:
            points = [[point[0] + offset, point[1] + offset]
                      for point in points]
        # 上下加
        elif n == 1:
            points = [[point[0], point[1] + offset] for point in points]
        # 左右加
        elif n == 2:
            points = [[point[0] + offset, point[1]] for point in points]
    elif indexs == 3:
        ind = random.sample(range(0, 4), 3)
        if n == 0:
            points[ind[0]][0] = points[ind[0]][0] + offset
            points[ind[0]][1] = points[ind[0]][1] + offset
            points[ind[1]][0] = points[ind[1]][0] + offset
            points[ind[1]][1] = points[ind[1]][1] + offset
            points[ind[2]][0] = points[ind[2]][0] + offset
            points[ind[2]][1] = points[ind[2]][1] + offset
        # 上下加
        elif n == 1:
            points[ind[0]][1] = points[ind[0]][1] + offset
            points[ind[1]][1] = points[ind[1]][1] + offset
            points[ind[2]][1] = points[ind[2]][1] + offset
        # 左右加
        elif n == 2:
            points[ind[0]][0] = points[ind[0]][0] + offset
            points[ind[1]][0] = points[ind[1]][0] + offset
            points[ind[2]][0] = points[ind[2]][0] + offset
    elif indexs == 2:
        ind = random.sample(range(0, 4), 2)
        if n == 0:
            points[ind[0]][0] = points[ind[0]][0] + offset
            points[ind[0]][1] = points[ind[0]][1] + offset
            points[ind[1]][0] = points[ind[1]][0] + offset
            points[ind[1]][1] = points[ind[1]][1] + offset
        # 上下加
        elif n == 1:
            points[ind[0]][1] = points[ind[0]][1] + offset
            points[ind[1]][1] = points[ind[1]][1] + offset
        # 左右加
        elif n == 2:
            points[ind[0]][0] = points[ind[0]][0] + offset
            points[ind[1]][0] = points[ind[1]][0] + offset
    elif indexs == 1:
        ind = random.sample(range(0, 4), 1)
        if n == 0:
            points[ind[0]][0] = points[ind[0]][0] + offset
            points[ind[0]][1] = points[ind[0]][1] + offset
        # 上下加
        elif n == 1:
            points[ind[0]][1] = points[ind[0]][1] + offset
        # 左右加
        elif n == 2:
            points[ind[0]][0] = points[ind[0]][0] + offset
    return points


def get_input(label, points, h, w):
    embeding = []
    p1, p2, p3, p4 = points
    point_array = np.array([p1[0] / w, p1[1] / h,
                            p2[0] / w, p2[1] / h,
                            p3[0] / w, p3[1] / h,
                            p4[0] / w, p4[1] / h] * 8)

    mean_embeding = model.infer_vector(list(str(label)))
    box = np.hstack((point_array, mean_embeding))
    return box


def get_output(group_id):
    if group_id in dic_label.keys():
        value = dic_label[group_id]
        onehot = np.eye(19)[value]
    else:
        onehot = np.eye(19)[0]
    onehot = [int(x) for x in onehot]
    return onehot


def get_json_label_seq2seq_model(path):
    boxs, onehots = [], []
    with open(path, 'rb') as f:
        data = json.loads(f.read())
        h = data["imageHeight"]
        w = data["imageWidth"]
    shapes = data["shapes"]
    shapes = aug(shapes)
    shapes = sorted(shapes, key=lambda x: (
        x["points"][0][1], x["points"][0][1]))
    for i, obj in enumerate(shapes):
        if i > 99:
            break
        label = obj["label"]
        points = obj["points"]
        group_id = obj["group_id"]
        if len(points) != 4:
            print(points, label)
            continue
        boxs.append(get_input(label, points, h, w))
        onehots.append(get_output(group_id))
    if len(shapes) < 100:
        for i in range(100 - len(shapes)):
            boxs.append(np.zeros((164)))
            onehots.append(np.zeros((19)))
    boxs = np.array(boxs)
    onehots = np.array(onehots)
    return boxs, onehots


def gen_seq2seq_model(paths, batch_size, seq_length):
    num = len(paths)
    while True:
        X = np.zeros((batch_size, seq_length, 164))
        Y = np.zeros((batch_size, seq_length, 19))
        for j in range(batch_size):
            i = np.random.randint(num)
            X[j], Y[j] = get_json_label_seq2seq_model(paths[i])
        yield X, Y


def get_json_test(path):
    X = np.zeros((1, 100, 164))
    Y = np.zeros((1, 100, 19))

    boxs, onehots = [], []
    with open(path, 'rb') as f:
        data = json.loads(f.read())
        h = data["imageHeight"]
        w = data["imageWidth"]
    shapes = data["shapes"]
    shapes = sorted(shapes, key=lambda x: (
        x["points"][0][1], x["points"][0][1]))
    for i, obj in enumerate(shapes):
        if i > 99:
            break
        label = obj["label"]
        points = obj["points"]
        group_id = obj["group_id"]
        if len(points) != 4:
            print(points, label)
            continue
        boxs.append(get_input(label, points, h, w))
        onehots.append(get_output(group_id))
    if len(shapes) < 100:
        for i in range(100 - len(shapes)):
            boxs.append(np.zeros((164)))
            onehots.append(np.zeros((19)))
    X[0], Y[0] = np.array(boxs), np.array(onehots)
    return X, Y


if __name__ == "__main__":
    import glob
    paths = glob.glob("datasets/train/*.json")
    txt = gen_seq2seq_model(paths, 8, 100)
    for i in range(32):
        a, b = next(txt)
        print(a.shape)
        print(b.shape)
