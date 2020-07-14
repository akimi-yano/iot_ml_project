import json

class Conf:
    def __init__(self, conf_path):
        with open(conf_path, "r") as f:
            self.conf = json.loads(''.join(f.readlines()))

    def __getitem__(self, key):
        return self.conf[key]
