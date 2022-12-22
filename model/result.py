class Result:

    """ Initialize """

    def __init__(self):
        self._loss = None
        self._acc = None
        self._roc = None
        self._loss_hist = None
        self._acc_hist = None

    """ Getters """

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def acc(self) -> float:
        return self._acc

    @property
    def roc(self) -> float:
        return self._roc

    @property
    def loss_hist(self) -> list:
        return self._loss_hist

    @property
    def acc_hist(self) -> list:
        return self._acc_hist

    """ Setters """

    @loss.setter
    def loss(self, loss: float):
        self._loss = loss

    @acc.setter
    def acc(self, acc: float):
        self._acc = acc

    @roc.setter
    def roc(self, roc: float):
        self._roc = roc

    @loss_hist.setter
    def loss_hist(self, loss_hist: list):
        self._loss_hist = loss_hist

    @acc_hist.setter
    def acc_hist(self, acc_hist: list):
        self._acc_hist = acc_hist
