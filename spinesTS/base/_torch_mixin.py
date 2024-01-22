import copy
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from spinesUtils.asserts import ParameterValuesAssert, augmented_isinstance, ParameterTypeAssert
from spinesUtils.logging import Logger

from spinesTS.metrics import WMAPELoss, RMSELoss
from spinesTS.utils import seed_everything, check_is_fitted


logger = Logger(with_time=False, name='')


@ParameterTypeAssert({
    'device': (str, None)
})
def detect_available_device(device='auto'):
    device = device or 'auto'

    if device != 'auto':
        device = device.lower()
        auto_selector = False
    else:
        auto_selector = True

    tpu_available = False

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()

        if device is not None:
            tpu_available = True

    except ImportError:
        if device == 'tpu':
            print("[ImportError]: torch_xla package is not installed."
                  "Consider run `python3 -m pip install torch_xla` in your terminal.")

    mps_available = False
    cuda_available = False
    cpu_available = True
    mps_use = False
    cuda_use = False
    cpu_use = False
    tpu_use = False

    if torch.backends.mps.is_available():
        mps_available = True

    if torch.cuda.is_available():
        cuda_available = True

    if tpu_available and (auto_selector or device == 'tpu'):
        device = xm.xla_device()
        tpu_use = True
    elif cuda_available and (auto_selector or device == 'cuda'):
        device = 'cuda:0' if torch.cuda.device_count() > 1 else 'cuda'
        cuda_use = True
    elif mps_available and (auto_selector or device == 'mps'):
        device = 'mps'
        mps_use = True
    elif cpu_available and (auto_selector or device == 'cpu'):
        device = 'cpu'
        cpu_use = True

    blank_length = lambda s: ' ' * 3 if s is True else ' ' * 2
    string_format = f"MPS  available: {mps_available}{blank_length(mps_available)}| MPS  use: {mps_use}\n" \
                    f"CUDA available: {cuda_available}{blank_length(cuda_available)}| CUDA use: {cuda_use}\n" \
                    f"TPU  available: {tpu_available}{blank_length(tpu_available)}| TPU  use: {tpu_use}\n" \
                    f"CPU  available: {cpu_available}{blank_length(cpu_available)}| CPU  use: {cpu_use}"

    return device, string_format


def clear_torch_cache(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


@ParameterValuesAssert({
    'name': lambda s: augmented_isinstance(s, (None, str))
})
def get_loss_func(name=None):
    """get loss function

    Parameters
    ----------
    name: str, name of loss function, default None

    Returns
    -------
    object, loss function.
    """
    names = {'huber': nn.HuberLoss(), 'mse': nn.MSELoss(), 'mae': nn.L1Loss(),
             'wmape': WMAPELoss(), 'rmse': RMSELoss()}
    if isinstance(name, str):
        name = name.lower()
        return names[name]
    else:
        return names['mae']


class TorchModelMixin:
    """Provide pytorch models common mixin class.

    This class make it easy to write code like this:
        ```python
        class Model(TorchModelMixin):
            def __init__(self, *args, **kwargs):
                # need to set random seed if you needed
                # need to set device which to put your tensor, default to cuda/cuda:0 if your gpu is available, else to cpu
                super(Model, self).__init__(seed=None, device=None)

                self.model, self.loss_fn, self.optimizer = self.call()  # implement your model architecture

            def call(self):
                # model = your_model_class()
                # loss_fn = your_loss_function()
                # optimizer = your_optimizer_function()
                # return model, loss_fn, optimizer
                pass

            def fit(self, X, y, *args, **kwargs):
                return super().fit(X, y, *args, **kwargs)

            # def metric(self, y_true, y_pred):
                 # your metric, default to mae(mean absolute error)
                 # if you want to use other metrics, you need to override this function

        # To fit something
        model = Model(args, kwargs)
        model.fit(X, y)

        # To predict something
        y_pred = model.predict(X)
        ```

    Parameters
    ----------
    seed : int or None, random seed
    device : str or None, device name
    loss_fn: str or None, loss function

    Returns
    -------
    None

    """

    def __init__(self, seed=None, device='auto', loss_fn='mae') -> None:
        self.training_logs = {
            'time_cost': [],
            'epochs': [],
            'batches': [],
            'lrs': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'current_p': []
        }

        seed_everything(seed)
        self.device, self.string_format = detect_available_device(device)

        clear_torch_cache(self.device)

        self.loss_fn_name = loss_fn
        self.loss_fn = get_loss_func(loss_fn)
        self.current_patience = 0

        self.__spinesTS_is_fitted__ = False

    def call(self, *args, **kwargs):
        """To implement the model architecture.

        """
        raise NotImplementedError("To implement a spinesTS.nn model class, you must implement a call function.")

    def fit(self,
            X,
            y,
            epochs=3000,
            batch_size='auto',
            eval_set=None,
            loss_type='down',
            metrics_name='score',
            monitor='val_loss',
            min_delta=0,
            patience=10,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.1,
            restore_best_weights=True,
            verbose=True,
            **lr_scheduler_kwargs
            ):
        """Fit your model.

        Parameters
        ----------
        X : torch.Tensor, training features
        y : torch.Tensor, training targets
        epochs : int, training epochs, default to 1000
        batch_size : str or int, 'auto' means to autofit, int means to specify the batch size
        eval_set : iterable object(tuple or list) of torch.Tensor, default to None
        loss_type : str, 'down' or 'rise', only be used if lr_scheduler='ReduceLROnPlateau'
            it means the way to set the learning rate scheduler to watch the loss value (down or rise)
        metrics_name : str, names your metrics, default to 'score'
        monitor : str, 'val_loss' or 'loss', quantity to be monitored,
        min_delta : minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement, default to 0
        patience : number of epochs with no improvement after which training will be stopped, default to 10
        lr_scheduler : learning rate scheduler name, one of ['ReduceLROnPlateau', 'CosineAnnealingLR',
            'CosineAnnealingWarmRestarts', None]
        lr_scheduler_patience :  number of epochs with no improvement after which learning rate will be reduced.
            For example, if patience = 2, then we will ignore the first 2 epochs with no improvement,
            and will only decrease the LR after the 3rd epoch  if the loss still hasn’t improved then, default: 10
        lr_factor : factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1
        restore_best_weights : Whether to restore model weights
                        from the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of training are used.
            If True, and if no epoch improves, training will run for patience epochs and restore weights from
                the best epoch in that set. Default to True.
        verbose : Whether to  displays messages, default to True
        **lr_scheduler_kwargs : torch.optim.lr_scheduler parameters

        Returns
        -------
        self

        """
        if verbose:
            logger.info('Information about the device used for computation:\n'+self.string_format)
            time.sleep(0.5)

        return self._fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            eval_set=eval_set,
            loss_type=loss_type,
            metrics_name=metrics_name,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            lr_scheduler=lr_scheduler,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            restore_best_weights=restore_best_weights,
            verbose=verbose,
            **lr_scheduler_kwargs
        )

    def predict(self, X):
        """
        X : torch.Tensor, tensor which to predict
        """
        check_is_fitted(self)
        self.model.eval()
        with torch.no_grad():
            X = torch.Tensor(X)
            X = self._move_to_device(X)
            pred = self.model(X)
        return pred.cpu().numpy()

    def _move_to_device(self, obj):
        obj = obj.to(self.device)
        return obj

    def metric(self, y_true, y_pred):
        """model metric"""
        return nn.functional.l1_loss(y_true, y_pred).item()

    def _get_batch_size(self, x, batch_size='auto'):
        if batch_size == 'auto':
            self._batch_size = 2 ** np.log2(len(x)) * 16
            self._batch_size = 5096 if self._batch_size > 5096 else self._batch_size
        else:
            assert isinstance(batch_size, int) and batch_size > 0
            self._batch_size = batch_size

    @staticmethod
    def _check_x_y_type(X, y):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        elif not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        elif not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)

        return X.float(), y.float()

    def data_loader(self, X, y):
        train_data = TensorDataset(X, y)
        train_loader = DataLoader(train_data, batch_size=self._batch_size, shuffle=False)

        return train_loader

    def train_on_one_epoch(
            self,
            dataloader,
            model,
            loss_fn,
            optimizer
    ):
        """Training function on one epoch
        If you want to override it, you just need to return two values,
        current loss on this epoch, average-accuracy on this epoch
        """
        model.train()  # set model to training mode
        train_batch = len(dataloader)
        train_loss_current, train_metric = 0, 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # compute error
            train_pred = model(x)
            train_loss = loss_fn(train_pred, y)
            optimizer.zero_grad(set_to_none=True)  # clear optimizer gradient

            # backward
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)

            optimizer.step()

            train_metric += self.metric(train_pred, y)
            train_loss_current += train_loss.item()  # 释放图形指针

        return train_loss_current / train_batch, train_metric / train_batch

    def test_on_one_epoch(
            self,
            dataloader,
            model,
            loss_fn
    ):
        """
        Test function on one epoch
        If you want to override it, you just need to return two values,
        current loss on this epoch, average-accuracy on this epoch

        """

        model.eval()  # set model to evaluate mode
        test_loss, test_metric, test_num_batches = 0, 0, len(dataloader)
        with torch.no_grad():  # with no gradient
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                pred = model(x)
                test_loss += loss_fn(y, pred).item()  # scalar
                # scalar
                test_metric += self.metric(y, pred)

        test_loss /= test_num_batches
        test_metric /= test_num_batches

        return test_loss, test_metric

    @ParameterTypeAssert({
        'loss_type': str,
        'min_delta': int,
        'patience': int,
        'restore_best_weights': bool
    })
    @ParameterValuesAssert({
        'loss_type': ('min', 'max')
    })
    def _early_stopping(
            self,
            loss,
            loss_type='min',
            min_delta=0,
            patience=10,
            restore_best_weights=True
    ):
        """
        loss type : min or max
        """
        if loss_type == 'max':
            loss = -loss

        if loss < (self.current_loss + min_delta):
            self.current_loss = loss
            if restore_best_weights:
                self.best_weight = copy.deepcopy(self.model.state_dict())
            self.current_patience = 0
        else:
            self.current_patience += 1

        if self.current_patience == patience:
            if restore_best_weights:
                self.model.load_state_dict(self.best_weight)
            return True

        return False

    def _get_lr_scheduler(
            self, mode=None,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.1,
            **lr_scheduler_kwargs
    ):
        if lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                patience=lr_scheduler_patience, factor=lr_factor, **lr_scheduler_kwargs
            )
        elif lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0,
                                                                   **lr_scheduler_kwargs)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2,
                                                                             **lr_scheduler_kwargs)
        elif lr_scheduler is None:
            return None
        else:
            raise KeyError(f"{lr_scheduler} is invalid.")

        return scheduler

    def print_training_log(self, init_lr, metrics_name, total_epochs):
        training_log_print = OrderedDict({
            "epoch_msg": "",
            "current_p_msg": "",
            "lr_msg": "",
            "loss_msg": "",
            "val_loss_msg": "",
            "time_msg": ""
        })
        training_log_print["epoch_msg"] = f"Epoch {self.training_logs['epochs'][-1] + 1:>1d}/" + \
                                          f"{total_epochs:>1d}  " + \
                                          f"\n\r{self.training_logs['batches'][-1]}/{self.training_logs['batches'][-1]}"

        training_log_print["lr_msg"] = f"[*lr: {self.training_logs['lrs'][-1]:4e}]" if len(
            str(self.training_logs['lrs'][-1])) <= 5 else f" [*lr: {self.training_logs['lrs'][-1]:>5}]"

        training_log_print["loss_msg"] = f"loss: {self.training_logs['train_loss'][-1]:>.4f} - " + \
                                         f"{metrics_name}: {self.training_logs['train_accuracy'][-1]:>.4f}"

        training_log_print[
            "val_loss_msg"
        ] = f"val_loss: {self.training_logs['test_loss'][-1]:>.4f} - " + \
            f"val_{metrics_name}: {self.training_logs['test_accuracy'][-1]:>.4f}"

        training_log_print["time_msg"] = \
            f"{self.training_logs['time_cost'][-1]:>.2f}s/epoch - " + \
            f"{self.training_logs['time_cost'][-1] / self.training_logs['batches'][-1]:>.3f}s/step"

        training_log_print["current_p_msg"] = f"p{self.training_logs['current_p'][-1]}"

        if round(self.training_logs['lrs'][-1], 5) == init_lr:
            del training_log_print['lr_msg']
        metric_string = ' - '.join([i for i in training_log_print.values()])

        return metric_string

    @staticmethod
    def _check_eval_set_params(eval_set):
        if isinstance(eval_set, tuple):
            assert len(eval_set) == 2
        elif isinstance(eval_set, list):
            assert len(eval_set[0]) == 2
            eval_set = eval_set[0]

        return eval_set

    def _fit(
            self,
            X,
            y,
            epochs=1000,
            batch_size='auto',
            eval_set=None,
            loss_type='min',
            metrics_name='score',
            monitor='val_loss',
            min_delta=0,
            patience=10,
            lr_scheduler='CosineAnnealingLR',
            lr_scheduler_patience=10,
            lr_factor=0.1,
            restore_best_weights=True,
            verbose=True,
            **lr_scheduler_kwargs
    ):
        """
        lr_scheduler: torch.optim.lr_scheduler class,
            only support to ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
        """

        assert eval_set is None or isinstance(eval_set, (list, tuple))
        assert monitor in ('train_loss', 'val_loss', None)

        init_lr = copy.deepcopy(self.learning_rate)

        self.model = self._move_to_device(self.model)
        X, y = self._check_x_y_type(X, y)

        eval_set = self._check_eval_set_params(eval_set)

        self._get_batch_size(X, batch_size=batch_size)
        train_dataloader = self.data_loader(X, y)
        test_dataloader = None

        if eval_set is not None:
            test_dataloader = self.data_loader(*self._check_x_y_type(eval_set[0], eval_set[1]))

        self.current_loss = np.finfo(np.float64).max - min_delta
        self.best_weight = copy.deepcopy(self.model.state_dict())

        batches = int(np.ceil(len(X) / self._batch_size))

        mode = loss_type

        scheduler = self._get_lr_scheduler(
            mode=mode, lr_scheduler=lr_scheduler,
            lr_scheduler_patience=lr_scheduler_patience,
            lr_factor=lr_factor,
            **lr_scheduler_kwargs)

        for epoch in range(epochs):
            tik = time.time()
            stop_state = False

            self.training_logs['epochs'].append(epoch)
            self.training_logs['batches'].append(batches)

            train_loss_current, train_acc = self.train_on_one_epoch(train_dataloader, model=self.model,
                                                                    loss_fn=self.loss_fn,
                                                                    optimizer=self.optimizer)

            if lr_scheduler:
                scheduler.step() if lr_scheduler != 'ReduceLROnPlateau' else scheduler.step(train_loss_current)

            self.training_logs['lrs'].append(round(float(self.optimizer.state_dict()['param_groups'][0]['lr']), 7))

            self.training_logs['train_loss'].append(train_loss_current)
            self.training_logs['train_accuracy'].append(train_acc)

            if monitor == 'train_loss':
                stop_state = self._early_stopping(train_loss_current, loss_type=loss_type,
                                                  min_delta=min_delta, patience=patience,
                                                  restore_best_weights=restore_best_weights)
            else:
                if test_dataloader:
                    test_loss, test_acc = self.test_on_one_epoch(test_dataloader, self.model, self.loss_fn)
                    stop_state = self._early_stopping(test_loss, loss_type=loss_type, min_delta=min_delta,
                                                      patience=patience, restore_best_weights=restore_best_weights)

                    self.training_logs['test_loss'].append(test_loss)
                    self.training_logs['test_accuracy'].append(test_acc)

            if monitor is not None and eval_set is not None:
                self.training_logs['current_p'].append(self.current_patience)

            tok = time.time()

            self.training_logs['time_cost'].append(tok - tik)

            if verbose:
                print(self.print_training_log(init_lr, metrics_name, epochs))

            if stop_state:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        self.__spinesTS_is_fitted__ = True
        return self

    def score(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X, y = torch.Tensor(X), torch.Tensor(y)
            X_gpu, _ = self._move_to_device(X), self._move_to_device(y)
            pred = self.model(X_gpu).cpu().numpy()

        return self.metric(y, pred)
