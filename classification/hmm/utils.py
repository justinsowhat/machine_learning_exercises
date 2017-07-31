class LabelIndexDict(object):

    def __init__(self):
        self.index_to_label = {}
        self.label_to_index = {}
        self.size = 0

    def size(self):
        return self.size

    def get_label_by_index(self, index):
        if index >= self.size:
            raise KeyError("Label index out of range")
        return self.index_to_label[index]

    def get_index_by_label(self, label):
        if label not in self.label_to_index:
            return self.size  # return the last index for unknown words or POS
        return self.label_to_index[label]

    def add(self, label):
        if label not in self.label_to_index:
            self.label_to_index[label] = self.size
            self.index_to_label[self.size] = label
            self.size += 1

    def set_index(self, label, index):
        self.label_to_index[label] = index