import gc
from datetime import datetime as dt

import numpy as np

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
        outputs_cnt = self.model.get_outputs_count()
        train_prev_losses = [float('inf') for _ in range(outputs_cnt)]
        val_best_losses = [float('inf') for _ in range(outputs_cnt)]
        val_prev_losses = [float('inf') for _ in range(outputs_cnt)]
        best_loss_epoch = None

        best_weights = last_weights = self.model.get_weights()
        reload_attempts = 0

        epoch = 1
        while epoch <= num_epochs:
            epoch_str = str(epoch).rjust(len(str(num_epochs)))
            print(f'[{dt.now()}]')
            print(f'Epoch {epoch_str}/{num_epochs}:')
            self.progress_tracker.message('epoch', {
                'current': epoch, 'total': num_epochs
            })
            self.progress_tracker.message('train_iteration', {
                'current': 0, 'total': len(self.train_dataset)
            })
            self.progress_tracker.message('val_iteration', {
                'current': 0, 'total': len(self.validation_dataset)
            })

            if self.optimizer is not None:
                print(f'  lr = {self.optimizer.lr}')

            ts = dt.now()

            train_losses = [0 for _ in range(outputs_cnt)]
            validation_losses = [0 for _ in range(outputs_cnt)]

            if self.show_progress_bar:
                def pb(iterable, *args, **kwargs):
                    return tqdm(iterable, *args, **kwargs)
            else:
                def pb(iterable, *args, **kwargs):
                    return iterable

            iters_cnt = len(self.train_dataset)
            for i in pb(range(iters_cnt), desc='Training', ascii=True):
                self.progress_tracker.reset()
                self.progress_tracker.message('training')

                X, ys = self.train_dataset.get(i)
                loss = self.model.train(X, ys)

                out_losses = loss['output_losses']
                for j in range(outputs_cnt):
                    train_losses[j] += out_losses[j]

                ps = None
                if self.save_pictures_func is not None:
                    ps = self.model.predict(X)
                    self.save_pictures_func(epoch, 'train', i, X, ys, ps)

                self.progress_tracker.message('train_iteration', {
                    'current': i + 1, 'total': iters_cnt
                })

                del X, ys, ps
                gc.collect()

            y_min, y_mean, y_max = None, None, None

            iters_cnt = len(self.validation_dataset)
            assert iters_cnt > 0, 'Validation dataset must have at least 1 element'
            for i in pb(range(iters_cnt), desc='Validating', ascii=True):
                self.progress_tracker.reset()
                self.progress_tracker.message('validating')

                X, ys = self.validation_dataset.get(i)
                loss = self.model.test(X, ys)

                out_losses = loss['output_losses']
                for j in range(outputs_cnt):
                    validation_losses[j] += out_losses[j]

                ps = None
                if self.save_pictures_func is not None or i == iters_cnt - 1:
                    self.progress_tracker.message('disable_status_update')
                    ps = self.model.predict(X)
                    self.progress_tracker.message('enable_status_update')
                if self.save_pictures_func is not None:
                    self.save_pictures_func(epoch, 'validation', i, X, ys, ps)

                if ps is not None:
                    y_min = ' '.join(f'{np.round(CP.asnumpy(CP.cp.min(y)), 3):<5}' for y in ys)
                    y_mean = ' '.join(f'{np.round(CP.asnumpy(CP.cp.mean(y)), 3):<5}' for y in ys)
                    y_max = ' '.join(f'{np.round(CP.asnumpy(CP.cp.max(y)), 3):<5}' for y in ys)
                    p_min = ' '.join(f'{np.round(CP.asnumpy(CP.cp.min(p)), 3):<5}' for p in ps)
                    p_mean = ' '.join(f'{np.round(CP.asnumpy(CP.cp.mean(p)), 3):<5}' for p in ps)
                    p_max = ' '.join(f'{np.round(CP.asnumpy(CP.cp.max(p)), 3):<5}' for p in ps)

                self.progress_tracker.message('val_iteration', {
                    'current': i + 1, 'total': iters_cnt
                })

                del X, ys, ps
                gc.collect()

            for i in range(outputs_cnt):
                train_losses[i] /= len(self.train_dataset)
                validation_losses[i] /= len(self.validation_dataset)

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
                    f'learning rate could be decreased to try avoiding NaN values')

            train_losses_str = [
                f'{train_loss: }' for train_loss in train_losses
            ]
            tmp_train_loss_diffs = [
                train_losses[i] - train_prev_losses[i]
                for i in range(outputs_cnt)
            ]
            avg_train_loss_diffs = np.mean(tmp_train_loss_diffs)
            train_loss_diffs = []
            for i in range(outputs_cnt):
                round_cnt = len(train_losses_str[i].split('.')[1])
                diff = round(tmp_train_loss_diffs[i], round_cnt)
                diff = f'{diff:+}'.ljust(len(train_losses_str[i]))
                train_loss_diffs.append(diff)
            train_losses_str = ' '.join(train_losses_str)
            train_loss_diffs = ' '.join(train_loss_diffs)
            train_prev_losses = train_losses

            validation_losses_str = [
                f'{validation_loss: }' for validation_loss in validation_losses
            ]
            tmp_val_loss_diffs = [
                validation_losses[i] - val_prev_losses[i]
                for i in range(outputs_cnt)
            ]
            avg_val_loss_diffs = np.mean(tmp_val_loss_diffs)
            val_loss_diffs = []
            for i in range(outputs_cnt):
                round_cnt = len(validation_losses_str[i].split('.')[1])
                diff = round(tmp_val_loss_diffs[i], round_cnt)
                diff = f'{diff:+}'.ljust(len(validation_losses_str[i]))
                val_loss_diffs.append(diff)
            validation_losses_str = ' '.join(validation_losses_str)
            val_loss_diffs = ' '.join(val_loss_diffs)
            val_prev_losses = validation_losses

            print(f'  Train loss:      {train_losses_str}')
            print(f'    Loss change:   {train_loss_diffs}')
            print(f'    Average loss change: {avg_train_loss_diffs:+}')
            print(f'  Validation loss: {validation_losses_str}')
            print(f'    Loss change:   {val_loss_diffs}')
            print(f'    Average loss change: {avg_val_loss_diffs:+}')
            print(f'  Statistics (y :: p):')
            print(f'    min:  {y_min} :: {p_min}')
            print(f'    mean: {y_mean} :: {p_mean}')
            print(f'    max:  {y_max} :: {p_max}')

            if np.mean(validation_losses) < np.mean(val_best_losses):
                best_weights = self.model.get_weights()
                if self.save_weights_func:
                    print(f'  Saving model weights')
                    self.save_weights_func()
                best_loss_epoch = epoch
                val_best_losses = validation_losses

            print(f'Time required: {dt.now() - ts}')
            print(f'\n')

            last_weights = self.model.get_weights()
            epoch += 1
            reload_attempts = 0

        return val_best_losses, best_loss_epoch
