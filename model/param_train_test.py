class TrainTestParam:

    """ Getters """

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def criterion(self):
        return self._criterion

    @property
    def train_data_loader(self):
        return self._train_data_loader

    @property
    def val_data_loader(self):
        return self._val_data_loader

    @property
    def test_data_loader(self):
        return self._test_data_loader

    @property
    def epochs(self):
        return self._epochs

    @property
    def device(self):
        return self._device

    @property
    def freeze_layers(self):
        return self._freeze_layers

    """ Setters """

    @model.setter
    def model(self, model):
        self._model = model

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    @train_data_loader.setter
    def train_data_loader(self, train_data_loader):
        self._train_data_loader = train_data_loader

    @val_data_loader.setter
    def val_data_loader(self, val_data_loader):
        self._val_data_loader = val_data_loader

    @test_data_loader.setter
    def test_data_loader(self, test_data_loader):
        self._test_data_loader = test_data_loader

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @device.setter
    def device(self, device):
        self._device = device

    @freeze_layers.setter
    def freeze_layers(self, freeze_layers):
        self._freeze_layers = freeze_layers
