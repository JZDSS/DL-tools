class Net(object):
    def __init__(self, name):
        self.name = name

    def build(self):
        raise NotImplementedError()

    def build_batch_norm_update_ops(self):
        raise NotImplementedError()
