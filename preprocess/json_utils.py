import json


def save_json(file_name, json_object):
    with open(file_name, 'w') as f:
        json.dump(json_object, f, indent=3)


def load_json(file_name):
    json_obj = None
    with open(file_name, 'r') as f:
        json_obj = json.load(f)
    return json_obj
