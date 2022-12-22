class LoadDataQuery:

    """ Initialize """

    def __init__(self, val_data_length,
                 train_neg_x_path,
                 train_pos_x_path,
                 test_neg_x_path,
                 test_pos_x_path,
                 train_batch_size,
                 val_batch_size,
                 test_batch_size):

        self._val_data_length = val_data_length
        self._train_neg_x_path = train_neg_x_path
        self._train_pos_x_path = train_pos_x_path
        self._test_neg_x_path = test_neg_x_path
        self._test_pos_x_path = test_pos_x_path
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._test_batch_size = test_batch_size

    """ Getters """

    @property
    def val_data_length(self):
        return self._val_data_length

    @property
    def train_neg_x_path(self):
        return self._train_neg_x_path

    @property
    def train_pos_x_path(self):
        return self._train_pos_x_path

    @property
    def test_neg_x_path(self):
        return self._test_neg_x_path

    @property
    def test_pos_x_path(self):
        return self._test_pos_x_path

    @property
    def train_batch_size(self):
        return self._train_batch_size

    @property
    def val_batch_size(self):
        return self._val_batch_size

    @property
    def test_batch_size(self):
        return self._test_batch_size
