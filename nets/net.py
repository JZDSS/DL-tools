class Net(object):
    def __init__(self, name):
        self.name = name

    def build(self, inputs):
        raise NotImplementedError()

    def get_update_ops(self):
        raise NotImplementedError()
