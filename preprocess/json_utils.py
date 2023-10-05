import json


def save_json(file_name, json_object):
    with open(file_name, 'w') as f:
        json.dump(json_object, f, indent=3)


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
