class AttentionResult:

    """ Initialize """

    def __init__(self, input, attention):
        self._input = input  # Input id
        self._attention = attention

    """ Getters """

    @property
    def input(self):
        return self._input

    @property
    def attention(self):
        return self._attention
