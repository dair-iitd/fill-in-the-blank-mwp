import json

class JSONLDataset:

    def __init__(self, filepath):

        self.filepath = filepath
        self.examples = []

        with open(filepath, 'r') as file:
            for l in file:
                self.examples.append(json.loads(l))

    def __getitem__(self, key):
        return self.examples[key]

    def __len__(self):
        return len(self.examples)

