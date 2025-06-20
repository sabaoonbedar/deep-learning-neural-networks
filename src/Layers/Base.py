
class BaseLayer:

    def __init__(self):
        # By default, layers are not trainable unless overridden
        self.trainable = False