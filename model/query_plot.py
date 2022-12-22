class PlotQuery:

    """ Initialize """

    def __init__(self, train_hist,
                 val_hist,
                 train_label,
                 val_label,
                 x_label,
                 y_label,
                 plot_title,
                 file_name):

        self._train_hist = train_hist
        self._val_hist = val_hist
        self._train_label = train_label
        self._val_label = val_label
        self._x_label = x_label
        self._y_label = y_label
        self._plot_title = plot_title
        self._file_name = file_name

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
    def plot_title(self):
        return self._plot_title

    @property
    def file_name(self):
        return self._file_name
