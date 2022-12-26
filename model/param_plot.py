class PlotParam:

    """ Getters """

    @property
    def train_hist(self):
        return self._train_hist

    @property
    def val_hist(self):
        return self._val_hist

    @property
    def train_label(self):
        return self._train_label

    @property
    def val_label(self):
        return self._val_label

    @property
    def x_label(self):
        return self._x_label

    @property
    def y_label(self):
        return self._y_label

    @property
    def tokens(self):
        return self._tokens

    @property
    def attention(self):
        return self._attention

    @property
    def plot_title(self):
        return self._plot_title

    @property
    def file_name(self):
        return self._file_name

    """ Setters """

    @train_hist.setter
    def train_hist(self, train_hist):
        self._train_hist = train_hist

    @val_hist.setter
    def val_hist(self, val_hist):
        self._val_hist = val_hist

    @train_label.setter
    def train_label(self, train_label):
        self._train_label = train_label

    @val_label.setter
    def val_label(self, val_label):
        self._val_label = val_label

    @x_label.setter
    def x_label(self, x_label):
        self._x_label = x_label

    @y_label.setter
    def y_label(self, y_label):
        self._y_label = y_label

    @tokens.setter
    def tokens(self, tokens):
        self._tokens = tokens

    @attention.setter
    def attention(self, attention):
        self._attention = attention

    @plot_title.setter
    def plot_title(self, plot_title):
        self._plot_title = plot_title

    @file_name.setter
    def file_name(self, file_name):
        self._file_name = file_name
