import glob
import json
import csv
import random
from config import Config


def jsonsline(paths):
    dataset_key = []
    dataset_other = []
    for path in paths:
        with open(path, 'rb') as f:
            data = json.loads(f.read())
            h = data["imageHeight"]
            w = data["imageWidth"]
            shapes = data["shapes"]

            for i, obj in enumerate(shapes):
                label = obj["label"].replace("\"","").replace("\'","")
                group_id = obj["group_id"]
                points = obj["points"]
                if len(points)==2:
                    p2 = [points[1][0],points[0][1]]
                    p3 = [points[0][0],points[1][1]]
                    location = points[0]+p2+points[1]+p3
                elif len(points)==4:
                    location = points[0]+points[1]+points[2]+points[3]
                else:
                    print(points)
                    continue

                location = ";".join([str(x) for x in location])
                if group_id:
                    dataset_key.append([label,group_id])
                else:
                    dataset_other.append([label,0])
    return dataset_key,dataset_other

def change_num(string):
    '''
    根据业务需求,随即替换数字
    '''
    ss = list(string)
    for i in range(len(ss)):
        if ss[i].isdigit():
            n = random.randint(0, 9)
            ss[i] = str(n)
    return ''.join(ss)


def write_line(datas,topath):
    with open(topath, "w") as f:
        write=csv.writer(f)
        write.writerow(["text","label"])
        for line in datas:
            line[0] = str(line[0]).replace(",", "").replace(" ", "")
            if line[1] is None:
                line[1] = 0
            write.writerow(line)


def aug_train(train_key,times):
    new = []
    for _ in range(times):
        for line in train_key:
            new.append([change_num(line[0]),line[1]])
    return new



if __name__ == "__main__":
    config = Config()
    train_jsons = glob.glob(config.train_jsons+"/*.json")
    test_jsons = glob.glob(config.test_jsons+"/*.json")

    train_key,train_other = jsonsline(train_jsons)
    test_key,test_other= jsonsline(test_jsons)

    new_train_key = aug_train(train_key,5)
    train_data = new_train_key+train_other
    test_data = test_key+test_other

    random.shuffle(train_data)
    random.shuffle(test_data)

    write_line(train_data,config.rnn_train_path)
    write_line(test_data,config.rnn_test_path)



