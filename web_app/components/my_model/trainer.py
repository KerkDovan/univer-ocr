import gc
from datetime import datetime as dt
from random import shuffle

import numpy as np

from tqdm import tqdm


class Losses:
    def __init__(self, model_names, outputs_cnts):
        self.model_names = model_names
        self.outputs_cnts = outputs_cnts
        self.train_prev_losses = self._new_losses(float('inf'))
        self.val_best_losses = self._new_losses(float('inf'))
        self.val_prev_losses = self._new_losses(float('inf'))
        self.train_losses = None
        self.val_losses = None
        self.best_loss_epoch = {name: 0 for name in self.model_names}

    def reset(self):
        self.train_losses = self._new_losses(0)
        self.val_losses = self._new_losses(0)

    def _new_losses(self, value):
        return {
            name: [value for _ in range(self.outputs_cnts[name])]
            for name in self.model_names
        }

    def get_better_weights(self, epoch):
        def cond(a, b):
            return np.mean(a) < np.mean(b) or not np.isnan(a) and np.isnan(b)
        result = [
            name for name in self.model_names
            if cond(self.val_losses[name], self.val_best_losses[name])
        ]
        for name in result:
            self.val_best_losses[name] = self.val_losses[name]
            self.best_loss_epoch[name] = epoch
        return result

    def next(self):
        self.train_prev_losses = self.train_losses
        self.val_prev_losses = self.val_losses

    def train(self, update):
        for name in self.model_names:
            out_losses = update[name]['output_losses']
            for i in range(self.outputs_cnts[name]):
                self.train_losses[name][i] += out_losses[i]

    def validation(self, update):
        for name in self.model_names:
            out_losses = update[name]['output_losses']
            for i in range(self.outputs_cnts[name]):
                self.val_losses[name][i] += out_losses[i]

    def normalize(self, train_dataset_size, validation_dataset_size):
        for name in self.model_names:
            for i in range(self.outputs_cnts[name]):
                self.train_losses[name][i] /= train_dataset_size
                self.val_losses[name][i] /= validation_dataset_size

    def print(self, left_margin=0):
        train_values, val_values = [], []
        tmp_train_diffs, tmp_val_diffs = [], []
        train_diffs, val_diffs = [], []

        for name in self.model_names:
            train_values.append([
                f'{train_loss: }'
                for train_loss in self.train_losses[name]
            ])
            val_values.append([
                f'{val_loss: }'
                for val_loss in self.val_losses[name]
            ])
            tmp_train_diffs.append([
                self.train_losses[name][i] - self.train_prev_losses[name][i]
                for i in range(self.outputs_cnts[name])
            ])
            train_diffs.append([f'{diff:+}' for diff in tmp_train_diffs[-1]])
            tmp_val_diffs.append([
                self.val_losses[name][i] - self.val_prev_losses[name][i]
                for i in range(self.outputs_cnts[name])
            ])
            val_diffs.append([f'{diff:+}' for diff in tmp_val_diffs[-1]])

        avg_train_diffs = [f'{np.mean(diffs):+}' for diffs in tmp_train_diffs]
        avg_val_diffs = [f'{np.mean(diffs):+}' for diffs in tmp_val_diffs]

        column_widths = []
        for i, name in enumerate(self.model_names):
            widths = []
            for j in range(self.outputs_cnts[name]):
                l1 = len(train_values[i][j])
                l2 = len(val_values[i][j])
                l3 = len(train_diffs[i][j])
                l4 = len(val_diffs[i][j])
                widths.append(max(l1, l2, l3, l4))
            l5 = len(avg_train_diffs[i])
            l6 = len(avg_val_diffs[i])
            column_widths.append(max(len(name), sum(widths), l5, l6))
            for arr in [train_values, val_values, train_diffs, val_diffs]:
                arr[i] = ' '.join(
                    str(val).ljust(widths[j])
                    for j, val in enumerate(arr[i]))

        result = [
            [name for name in self.model_names],
            train_values, train_diffs, avg_train_diffs,
            val_values, val_diffs, avg_val_diffs]
        for i in range(len(result)):
            result[i] = ' | '.join(
                val.ljust(column_widths[i]) for i, val in enumerate(result[i]))

        lm = ' ' * left_margin
        print(lm + f'Models:            {result[0]}')
        print(lm + f'Train loss:        {result[1]}')
        print(lm + f'  Loss change:     {result[2]}')
        print(lm + f'  Avg loss change: {result[3]}')
        print(lm + f'Validation loss:   {result[4]}')
        print(lm + f'  Loss change:     {result[5]}')
        print(lm + f'  Avg loss change: {result[6]}')


class Trainer:
    def __init__(self, model_system, make_context_func,
                 models, train_dataset, validation_dataset,
                 progress_tracker, show_progress_bar=False,
                 optimizer=None, learning_rate_step=0.995,
                 save_weights_func=None, save_pictures_func=None):
        self.model_system = model_system
        self.make_context_func = make_context_func
        self.models = models
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.progress_tracker = progress_tracker
        self.show_progress_bar = show_progress_bar
        self.optimizer = optimizer
        self.learning_rate_step = learning_rate_step
        self.save_weights_func = save_weights_func
        self.save_pictures_func = save_pictures_func

    def train(self, num_epochs):
        if self.show_progress_bar:
            def pb(iterable, *args, **kwargs):
                return tqdm(iterable, *args, **kwargs)
        else:
            def pb(iterable, *args, **kwargs):
                return iterable

        model_names = list(self.models.keys())
        outputs_cnts = {
            name: model.get_outputs_count()
            for name, model in self.models.items()
        }
        losses = Losses(model_names, outputs_cnts)

        print('Precomputing losses')
        ts = dt.now()
        losses.reset()
        for i in pb(range(len(self.validation_dataset)), desc='Precomputing', ascii=True):
            context = self.make_context_func(
                self.validation_dataset.get, (i,))
            self.model_system.test(context)
            losses.validation(context['losses'])
            if self.save_pictures_func is not None:
                self.save_pictures_func(0, 'precomputing', i, context)
            del context
            gc.collect()
        losses.print(left_margin=2)
        losses.next()
        print(f'Time required: {dt.now() - ts}')
        print(f'\n')

        def get_weights():
            return {
                name: weights
                for model in self.models.values()
                for name, weights in model.get_weights().items()
            }

        best_weights = last_weights = get_weights()
        reload_attempts = 0

        train_random_order = list(range(len(self.train_dataset)))
        validation_random_order = list(range(len(self.validation_dataset)))

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

            losses.reset()

            shuffle(train_random_order)
            iters_cnt = len(self.train_dataset)
            for i in pb(range(iters_cnt), desc='Training', ascii=True):
                self.progress_tracker.reset()
                self.progress_tracker.message('training')

                context = self.make_context_func(
                    self.train_dataset.get, (train_random_order[i],))
                self.model_system.train(context)
                losses.train(context['losses'])

                if self.save_pictures_func is not None:
                    self.save_pictures_func(epoch, 'train', i, context)

                self.progress_tracker.message('train_iteration', {
                    'current': i + 1, 'total': iters_cnt
                })

                del context
                gc.collect()

            shuffle(validation_random_order)
            iters_cnt = len(self.validation_dataset)
            assert iters_cnt > 0, 'Validation dataset must have at least 1 element'
            for i in pb(range(iters_cnt), desc='Validating', ascii=True):
                self.progress_tracker.reset()
                self.progress_tracker.message('validating')

                context = self.make_context_func(
                    self.validation_dataset.get, (validation_random_order[i],))
                self.model_system.test(context)
                losses.validation(context['losses'])

                if self.save_pictures_func is not None:
                    self.save_pictures_func(epoch, 'validation', i, context)

                self.progress_tracker.message('val_iteration', {
                    'current': i + 1, 'total': iters_cnt
                })

                del context
                gc.collect()

            losses.normalize(len(self.train_dataset), len(self.validation_dataset))

            if self.optimizer is not None:
                reload_attempts += 1
                self.optimizer.lr *= self.learning_rate_step ** reload_attempts

                if any(model.nan_weights() for model in self.models.values()):
                    if reload_attempts < 10:
                        print(f'NaN value found in weights, loading last weights\n')
                        for model in self.models.values():
                            model.set_weights(last_weights)
                    else:
                        print(f'Too many attempts, loading last best weights\n')
                        for model in self.models.values():
                            model.set_weights(best_weights)
                        reload_attempts = 0
                    continue

            elif any(model.nan_weights() for model in self.models.values()):
                raise ValueError(
                    f'NaN value found in weights, but no optimizer provided. '
                    f'Provide optimizer and learning_rate_step, so '
                    f'learning rate could be decreased to try avoiding NaN values')

            losses.print(left_margin=2)

            better_weights = losses.get_better_weights(epoch)
            if any(better_weights):
                if self.save_weights_func:
                    print(f'  Saving weights for ' + ', '.join(better_weights))
                    self.save_weights_func(better_weights)

            print(f'Time required: {dt.now() - ts}')
            print(f'\n')

            last_weights = get_weights()
            epoch += 1
            reload_attempts = 0
            losses.next()

        return losses.val_best_losses, losses.best_loss_epoch
