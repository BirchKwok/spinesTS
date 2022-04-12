import copy
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from spinesTS.metrics import mean_absolute_error
from spinesTS.utils import torch_summary


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = True if torch.cuda.is_available() else False


class TorchModelMixin:
    """Provide pytorch models common mixin class

    this class make it easy to write code like this:
        ```python
        class Model(TorchModelMixin):
            def __init__(self, *args, **kwargs):
                # if you want to set random seed
                from spinesTS.utils import seed_everything
                seed_everything(seed)

                self.call()  # implement your model architecture

            def call(self):
                # model = your_model_class()
                # loss_fn = your_loss_function()
                # optimizer = your_optimizer_function()
                # return model, loss_fn, optimizer
                pass

            def metric(self, y_true, y_pred):
                # implement your metric function
                pass

            def fit(self, *args, **kwargs):
                return self._fit(*args, **kwargs)

            def predict(self, *args, **kwargs):
                return self._predict(*args, **kwargs)

            def metric(self, y_true, y_pred):
                # your metric, default to mae
                # if you want to use mae, please ignore this implement
                pass

        # To fit something
        model = Model(args, kwargs)
        model.fit(X, y)

        # To predict something
        y_pred = model.predict(X)
        ```
    """

    def call(self, *args, **kwargs):
        raise NotImplementedError("To implement a spinesTS.nn model class, you must be implement a call function.")

    def fit(self, *args, **kwargs):
        raise NotImplementedError("To implement a spinesTS.nn model class, you must be implement a fit function.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("To implement a spinesTS.nn model class, you must be implement a predict function.")

    def move_to_device(self, obj, d=DEVICE):
        obj = obj.to(d)
        return obj

    def metric(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def _get_batch_size(self, x, batch_size='auto'):
        if batch_size == 'auto':
            self._batch_size = 32 if len(x) < 800 else len(x) // 40
        else:
            assert isinstance(batch_size, int) and batch_size > 0
            self._batch_size = batch_size

    def train(
            self,
            X,
            y,
            model,
            loss_fn,
            optimizer,
            batch_size
    ):
        train_data = TensorDataset(X, y)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=PIN_MEMORY)

        model.train()  # 将模型设置为训练模式
        train_batch = len(train_loader)
        train_loss_current, train_acc = 0, 0
        for batch_ndx, (x_, y_) in enumerate(train_loader):
            x_, y_ = x_.to(DEVICE), y_.to(DEVICE)
            # 计算预测误差
            train_pred = model(x_)
            train_loss = loss_fn(train_pred, y_)

            # 反向传播
            optimizer.zero_grad()  # 先将优化器中的累计梯度置空
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)

            optimizer.step()  # 对当前步骤执行优化

            train_acc += self.metric(y_.detach().cpu().numpy(), np.squeeze(train_pred.detach().cpu().numpy()))
            train_loss_current = train_loss.item()

            # train_batch += 1

        return train_loss_current, train_acc / train_batch

    def test(
            self,
            X,
            y,
            model,
            loss_fn,
            batch_size
    ):
        X_t, y_t = torch.Tensor(X), torch.Tensor(y)

        test_data = TensorDataset(X_t, y_t)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=PIN_MEMORY)

        model.eval()  # 将模型设置为预测模式
        test_loss, test_acc, test_num_batches = 0, 0, len(test_loader)
        with torch.no_grad():  # 测试环节不用计算梯度，减少计算量
            for batch_ndx, (x_, y_) in enumerate(test_loader):
                x_, y_ = x_.to(DEVICE), y_.to(DEVICE)
                pred = model(x_)
                test_loss += loss_fn(pred, y_).item()  # 返回一个标量，表示在测试集上的损失
                # 返回一个标量，表示在测试集上的准确与否
                test_acc += self.metric(y_.cpu().numpy(), np.squeeze(pred.cpu().numpy()))

        test_loss /= test_num_batches
        test_acc /= test_num_batches

        return test_loss, test_acc

    def _early_stopping(
            self,
            loss,
            loss_type='down',
            min_delta=0,
            patience=10,
            restore_best_weights=True
    ):
        """
        loss type : rise or down
        """
        assert loss_type in ('down', 'rise')

        if loss_type == 'rise':
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

    def _fit(
            self,
            X,
            y,
            epochs=1000,
            batch_size='auto',
            eval_set=None,
            loss_type='down',
            metrics_name='score',
            monitor='val_loss',
            min_delta=0,
            patience=10,
            lr_scheduler='ReduceLROnPlateau',
            lr_scheduler_patience=10,
            lr_factor=0.5,
            restore_best_weights=True,
            verbose=True,
            **lr_scheduler_kwargs
    ):
        """
        lr_Sceduler: torch.optim.lr_scheduler class, 
            only support to ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
        """

        assert eval_set is None or isinstance(eval_set, (list, tuple))
        assert monitor in ('loss', 'val_loss', None)

        if isinstance(eval_set, tuple):
            assert len(eval_set) == 2
        elif isinstance(eval_set, list):
            assert len(eval_set[0]) == 2
            eval_set = eval_set[0]

        self.model = self.move_to_device(self.model)
        X, y = torch.Tensor(X), torch.Tensor(y)

        self._get_batch_size(X, batch_size=batch_size)

        self.current_patience = 0
        self.current_loss = np.finfo(np.float64).max - min_delta
        self.best_weight = copy.deepcopy(self.model.state_dict())
        batches = int(np.ceil(len(X) / self._batch_size))

        if lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                   mode='min' if loss_type == 'down' else 'max',
                                                                   patience=lr_scheduler_patience, factor=lr_factor, **lr_scheduler_kwargs)
        elif lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0, **lr_scheduler_kwargs)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5,T_mult=2, **lr_scheduler_kwargs)
        elif lr_scheduler == None:
            pass
        else:
            raise KeyError(f"{lr_scheduler} is invalid.")

        for epoch in range(epochs):
            tik = time.time()
            stop_state = False

            metric_string = f"Epoch {epoch + 1:>1d}/{epochs:>1d} \n " \
                            f"\r{batches}/{batches} -"

            train_loss_current, train_acc = self.train(X, y, model=self.model, loss_fn=self.loss_fn,
                                                       optimizer=self.optimizer, batch_size=self._batch_size)

            if lr_scheduler:
                last_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                scheduler.step() if lr_scheduler != 'ReduceLROnPlateau' else scheduler.step(train_loss_current)
                if last_lr != self.optimizer.state_dict()['param_groups'][0]['lr']:
                    last_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                    metric_string += f" [*lr: {last_lr:>8}] -" if len(str(last_lr)) <= 8 else f" [*lr: {last_lr:>8}] -\n"

            metric_string += f" loss: {train_loss_current:>.4f} - {metrics_name}: {train_acc:>.4f} -"

            if monitor == 'loss':
                stop_state = self._early_stopping(train_loss_current, loss_type=loss_type,
                                                  min_delta=min_delta, patience=patience,
                                                  restore_best_weights=restore_best_weights)
            else:
                if eval_set is not None:
                    X_t, y_t = torch.Tensor(eval_set[0]), torch.Tensor(eval_set[1])
                    test_loss, test_acc = self.test(X_t, y_t, self.model, self.loss_fn, self._batch_size)
                    stop_state = self._early_stopping(test_loss, loss_type=loss_type, min_delta=min_delta,
                                                      patience=patience, restore_best_weights=restore_best_weights)

                    metric_string += f" val_loss: {test_loss:>.4f} - val_{metrics_name}: {test_acc:>.4f} -"

            if monitor is not None:
                ms_list = metric_string.split('-')
                ms_list.insert(1, f" p{self.current_patience} ")
                metric_string = '-'.join(ms_list)

            if verbose:
                tok = time.time()
                metric_string += f" {tok - tik:>.2f}s/epoch - {(tok - tik) / batches:>.3f}s/step"
                print(metric_string)

            if stop_state:
                break

        return self

    def _predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.Tensor(X)
            X = self.move_to_device(X)
            pred = self.model(X)
        return pred.cpu().numpy()

    def score(self, X, y):
        self.model.eval()
        with torch.no_grad():
            X, y = torch.Tensor(X), torch.Tensor(y)
            X_cuda, y_cuda = self.move_to_device(X), self.move_to_device(y)
            pred = self.model(X_cuda).cpu().numpy()

        return self.metric(y, pred)

    def summary(self):
        assert self.model is not None, "model must be not None."
        if self.model is not None:
            torch_summary(self.model, input_shape=(self.in_features,))
