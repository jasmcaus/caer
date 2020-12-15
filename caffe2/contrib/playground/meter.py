




from abc import abstractmethod


class Meter(object):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def Reset(self):
        pass

    @abstractmethod
    def Add(self):
        pass

    @abstractmethod
    def Compute(self):
        pass
