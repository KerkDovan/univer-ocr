import gc


class Trainer:
    def __init__(self, model, train_data_getter, test_data_getter, progress_tracker):
        self.model = model
        self.train_data_getter = train_data_getter
        self.test_data_getter = test_data_getter
        self.progress_tracker = progress_tracker

    def train_once(self, num_epochs=3):
        self.progress_tracker.message('generating_data')
        train_X, train_y = self.train_data_getter()

        prev_loss = 1e8

        for epoch in range(1, num_epochs + 1):
            self.progress_tracker.reset()
            self.progress_tracker.message('training')

            epoch_str = str(epoch).rjust(len(str(num_epochs)))
            print(f'Epoch {epoch_str}/{num_epochs}:', end=' ')

            train_loss = self.model.train(train_X, train_y)

            out_loss = train_loss['output_losses'][0]
            loss_diff = out_loss - prev_loss
            prev_loss = out_loss
            sign = '+' if loss_diff > 0 else ''

            print(f'Train loss: {train_loss}; {sign}{loss_diff}')

            gc.collect()

        self.progress_tracker.reset()
        self.progress_tracker.message('generating_data')

        test_X, test_y = self.test_data_getter()
        self.progress_tracker.message('testing')
        test_loss = self.model.test(test_X, test_y)

        print('-' * 60)
        print(f'Test loss:  {test_loss}')

        del train_X, train_y
        del test_X, test_y
        gc.collect()

        return train_loss, test_loss
