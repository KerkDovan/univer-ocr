import gc
from datetime import datetime as dt

from tqdm import tqdm

from ..nn.gpu import CP


class Trainer:
    def __init__(self, model, train_dataset, validation_dataset,
                 progress_tracker, show_progress_bar=False,
                 optimizer=None, learning_rate_step=0.995,
                 save_weights_func=None, save_pictures_func=None):
        self.model = model
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.progress_tracker = progress_tracker
        self.show_progress_bar = show_progress_bar
        self.optimizer = optimizer
        self.learning_rate_step = learning_rate_step
        self.save_weights_func = save_weights_func
        self.save_pictures_func = save_pictures_func

    def train(self, num_epochs):
        train_prev_loss = val_best_loss = val_prev_loss = float('inf')
        best_loss_epoch = None

        best_weights = last_weights = self.model.get_weights()
        reload_attempts = 0

        epoch = 1
        while epoch <= num_epochs:
            epoch_str = str(epoch).rjust(len(str(num_epochs)))
            print(f'[{dt.now()}]')
            print(f'Epoch {epoch_str}/{num_epochs}:')
            if self.optimizer is not None:
                print(f'  lr = {self.optimizer.lr}')

            ts = dt.now()

            train_loss = validation_loss = 0

            if self.show_progress_bar:
                def pb(iterable, *args, **kwargs):
                    return tqdm(iterable, *args, **kwargs)
            else:
                def pb(iterable, *args, **kwargs):
                    return iterable

            for i in pb(range(len(self.train_dataset)), desc='Training', ascii=True):
                self.progress_tracker.reset()
                self.progress_tracker.message('training')

                X, y = self.train_dataset.get(i)
                loss = self.model.train(X, y)

                out_loss = loss['output_losses'][0]
                train_loss += out_loss

                p = None
                if self.save_pictures_func is not None:
                    p = self.model.predict(X)
                    self.save_pictures_func(epoch, 'train', i, X, y, p)

                del X, y, p
                gc.collect()

            y_min, y_mean, y_max = None, None, None

            for i in pb(range(len(self.validation_dataset)), desc='Validating', ascii=True):
                self.progress_tracker.reset()
                self.progress_tracker.message('validating')

                X, y = self.validation_dataset.get(i)
                loss = self.model.test(X, y)

                out_loss = loss['output_losses'][0]
                validation_loss += out_loss

                p = None
                if self.save_pictures_func is not None:
                    p = self.model.predict(X)
                    self.save_pictures_func(epoch, 'validation', i, X, y, p)

                y_min = CP.asnumpy(CP.cp.min(y))
                y_mean = CP.asnumpy(CP.cp.mean(y))
                y_max = CP.asnumpy(CP.cp.max(y))

                p_min = CP.asnumpy(CP.cp.min(p[0]))
                p_mean = CP.asnumpy(CP.cp.mean(p[0]))
                p_max = CP.asnumpy(CP.cp.max(p[0]))

                del X, y, p
                gc.collect()

            train_loss /= len(self.train_dataset)
            validation_loss /= len(self.validation_dataset)

            if self.optimizer is not None:
                self.optimizer.lr *= self.learning_rate_step

                if self.model.nan_weights():
                    if reload_attempts < 10:
                        print(f'NaN value found in weights, loading last weights\n')
                        self.model.set_weights(last_weights)
                        reload_attempts += 1
                    else:
                        print(f'Too many attempts, loading last best weights\n')
                        self.model.set_weights(best_weights)
                    continue

            elif self.model.nan_weights():
                raise ValueError(
                    f'NaN value found in weights, but no optimizer provided. '
                    f'Provide optimizer and learning_rate_step, so '
                    f'learning rate could be decreased to avoid NaN values')

            train_loss_diff = train_loss - train_prev_loss
            train_prev_loss = train_loss

            val_loss_diff = validation_loss - val_prev_loss
            val_prev_loss = validation_loss

            print(f'  Train loss:       {train_loss}')
            print(f'    Loss change:   {train_loss_diff:+}')
            print(f'  Validation loss:  {validation_loss}')
            print(f'    Loss change:   {val_loss_diff:+}')
            print(f'  Statistics (y :: p):')
            print(f'    min:  {y_min:<5.4} :: {p_min:<5.4}')
            print(f'    mean: {y_mean:<5.4} :: {p_mean:<5.4}')
            print(f'    max:  {y_max:<5.4} :: {p_max:<5.4}')

            if validation_loss < val_best_loss:
                best_weights = self.model.get_weights()
                if self.save_weights_func:
                    print(f'  Saving model weights')
                    self.save_weights_func()
                best_loss_epoch = epoch
                val_best_loss = validation_loss

            print(f'Time required: {dt.now() - ts}')
            print(f'\n')

            last_weights = self.model.get_weights()
            epoch += 1
            reload_attempts = 0

        return val_best_loss, best_loss_epoch
