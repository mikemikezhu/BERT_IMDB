from model.result_attention import AttentionResult


class Result:

    """ Initialize """

    def __init__(self):
        self._loss = None
        self._acc = None
        self._roc = None
        self._loss_hist = None
        self._acc_hist = None
        self._tp_attention = None  # True positive
        self._tn_attention = None  # True negative
        self._fp_attention = None  # False positive
        self._fn_attention = None  # False negative

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

    @property
    def tp_attention(self) -> AttentionResult:
        return self._tp_attention

    @property
    def tn_attention(self) -> AttentionResult:
        return self._tn_attention

    @property
    def fp_attention(self) -> AttentionResult:
        return self._fp_attention

    @property
    def fn_attention(self) -> AttentionResult:
        return self._fn_attention

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

    @tp_attention.setter
    def tp_attention(self, tp_attention: AttentionResult):
        self._tp_attention = tp_attention

    @tn_attention.setter
    def tn_attention(self, tn_attention: AttentionResult):
        self._tn_attention = tn_attention

    @fp_attention.setter
    def fp_attention(self, fp_attention: AttentionResult):
        self._fp_attention = fp_attention

    @fn_attention.setter
    def fn_attention(self, fn_attention: AttentionResult):
        self._fn_attention = fn_attention
