class Trainer:
    def __init__(self, model, train_data_getter, test_data_getter, progress_tracker):
        self.model = model
        self.train_data_getter = train_data_getter
        self.test_data_getter = test_data_getter
        self.progress_tracker = progress_tracker

    def train_once(self):
        self.progress_tracker.message('generating_data')
        train_X, train_y = self.train_data_getter()

        self.progress_tracker.reset()
        self.progress_tracker.message('training')
        train_loss = self.model.train(train_X, train_y)

        self.progress_tracker.message('generating_data')
        test_X, test_y = self.test_data_getter()

        self.progress_tracker.reset()
        self.progress_tracker.message('testing')
        test_loss = self.model.test(test_X, test_y)

        return train_loss, test_loss
