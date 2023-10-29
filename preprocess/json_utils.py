import json
import pickle as pkl


def save_json(file_name, json_object):
    with open(file_name, 'w') as f:
        json.dump(json_object, f, indent=3)


def save_json_l(file_name, json_l_object):
    with open(file_name, 'w') as f:
        for obj in json_l_object:
            json.dump(obj, f)
            f.write('\n')


def load_json(file_name):
    with open(file_name, 'r') as f:
        json_obj = json.load(f)
    return json_obj


def load_json_l_qa(file_name):
    json_obj = {}
    i = 0
    with open(file_name, 'r') as f:
        for qa in f.readlines():
            temp = json.loads(qa)
            json_obj[i] = temp
            i += 1
    return json_obj


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        pkl_obj = pkl.load(f)
    return pkl_obj


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
