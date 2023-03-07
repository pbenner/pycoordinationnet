import torch
import pytorch_lightning as pl

from typing import Any, Dict, List, Optional
from monty.serialization import dumpfn

from sklearn.model_selection import KFold

from .model_estimator        import Lamb
from .model_transformer      import ModelTransformer
from .model_transformer_data import CryturesData, CryturesLoader

## ----------------------------------------------------------------------------

import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

## ----------------------------------------------------------------------------

class LitMetricTracker(pl.callbacks.Callback):
  def __init__(self):
    self.val_error_batch   = []
    self.val_error         = []
    self.train_error_batch = []
    self.train_error       = []

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self.train_error_batch.append(outputs['loss'].item())

  def on_train_epoch_end(self, *args, **kwargs):
    self.train_error.append(torch.mean(torch.tensor(self.train_error_batch)))
    self.train_error_batch = []

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    self.val_error_batch.append(outputs['val_loss'].item())

  def on_validation_epoch_end(self, trainer, pl_module):
    self.val_error.append(torch.mean(torch.tensor(self.val_error_batch)))
    self.val_error_batch = []

## ----------------------------------------------------------------------------

class LitProgressBar(pl.callbacks.progress.TQDMProgressBar):
    # Disable validation progress bar
    def on_validation_start(self, trainer, pl_module):
        pass
    def on_validation_end(self, trainer, pl_module):
        pass

## ----------------------------------------------------------------------------

class VerboseOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer):
        self.optimizer         = optimizer
        self.param_groups_copy = None
        self._copy_parameters()

    def _copy_parameters(self):
        self.param_groups_copy = []
        for param_group in self.optimizer.param_groups:
            param_group_copy = []
            for parameters in param_group['params']:
                param_group_copy.append(torch.clone(parameters.data))
            self.param_groups_copy.append(param_group_copy)

    def _print_difference(self):
        delta_min = torch.inf
        delta_max = 0.0
        delta_sum = 0.0
        delta_n   = 1.0

        for i, param_group in enumerate(self.optimizer.param_groups):
            for j, parameters in enumerate(param_group['params']):
                delta = torch.sum(torch.abs(self.param_groups_copy[i][j] - parameters.data)).item()

                if delta_min > delta:
                    delta_min = delta
                if delta_max < delta:
                    delta_max = delta

                delta_sum += delta
                delta_n   += 1.0

                print(f'update ({i},{j}): {delta:15.10f}')

        print(f'update max :', delta_max)
        print(f'update min :', delta_min)
        print(f'update mean:', delta_sum / delta_n)
        print()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self, *args, **kwargs):
        return self.optimizer.state_dict(*args, **kwargs)

    def step(self, closure=None):
        self._copy_parameters()
        self.optimizer.step(closure=closure)
        self._print_difference()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

## ----------------------------------------------------------------------------

class LitCryturesData(pl.LightningDataModule):
    def __init__(self, filename, model_config, n_splits = 1, val_size = 0.2, batch_size = 32, num_workers = 2, shuffle = True, random_state = 42):
        super().__init__()
        self.model_config = model_config
        self.num_workers  = num_workers
        self.val_size     = val_size
        self.batch_size   = batch_size
        self.data         = CryturesData(filename)
        # Setup k-fold cross-validation if n_splits > 1
        self.n_splits     = n_splits
        if n_splits > 1:
            self.splits   = list(KFold(n_splits, shuffle = shuffle, random_state = random_state).split(self.data.X, self.data.y))

    def setup_fold(self, k):
        if self.n_splits < 2:
            raise ValueError(f'k-fold cross-validation is not available, because n_splits is set to {self.n_splits}')
        self.k = k

    def setup(self, stage: Optional[str] = None):
        # Use pre-defined indices to split data into train and test
        if self.n_splits > 1:
            train_index, test_index = self.splits[self.k]
            data_train      = torch.utils.data.Subset(self.data, train_index)
            data_test       = torch.utils.data.Subset(self.data,  test_index)
        else:
            data_train = self.data
            data_test  = None
        # Take a piece of the training data for validation
        data_train, data_val = torch.utils.data.random_split(data_train, [1.0 - self.val_size, self.val_size])
        # Save result
        self.data_train = data_train
        self.data_test  = data_test
        self.data_val   = data_val

    def train_dataloader(self):
        return CryturesLoader(self.data_train, self.model_config, batch_size = self.batch_size, num_workers = self.num_workers)

    def val_dataloader(self):
        return CryturesLoader(self.data_val,   self.model_config, batch_size = self.batch_size, num_workers = self.num_workers)

    def test_dataloader(self):
        if self.n_splits < 2:
            raise ValueError('No test data available, use n_splits > 1')

        return CryturesLoader(self.data_test,  self.model_config, batch_size = self.batch_size, num_workers = self.num_workers)

## ----------------------------------------------------------------------------

class LitModelTransformer(pl.LightningModule):
    def __init__(self,
                 # Learning rate
                 lr = 1e-3, lr_groups = {},
                 # Weight decay
                 weight_decay = 0.0, weight_decay_groups = {},
                 # Other hyperparameters
                 betas = (0.9, 0.999), factor = 0.8, patience = 5,
                 # Optimizer and scheduler selection
                 scheduler = None, optimizer = 'Adam', optimizer_verbose = False, **kwargs):
        super().__init__()
        # Save all hyperparameters to `hparams` (e.g. lr)
        self.save_hyperparameters()
        self.loss              = torch.nn.L1Loss()
        self.train_loss        = []
        self.val_loss          = []
        self.optimizer         = optimizer
        self.optimizer_verbose = optimizer_verbose
        self.scheduler         = scheduler
        self.model             = ModelTransformer(**kwargs)
    
    def configure_optimizers(self):
        # Get learning rates
        lr        = self.hparams['lr']
        lr_groups = self.hparams['lr_groups']
        # Get weight_decay parameters
        weight_decay        = self.hparams['weight_decay']
        weight_decay_groups = self.hparams['weight_decay_groups']

        # Get parameter groups
        parameter_groups = []
        for name, params in self.model.parameters_grouped().items():
            group = {'params': params}

            if name in lr_groups:
                group['lr'] = lr_groups[name]
            if name in weight_decay_groups:
                group['weight_decay'] = weight_decay_groups[name]

            parameter_groups.append(group)

        # Initialize optimizer
        if   self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameter_groups, lr=lr, weight_decay=weight_decay, betas=self.hparams['betas'])
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(parameter_groups, lr=lr, weight_decay=weight_decay, betas=self.hparams['betas'])
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameter_groups, lr=lr, weight_decay=weight_decay)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(parameter_groups, lr=lr, weight_decay=weight_decay)
        elif self.optimizer == 'Lamb':
            optimizer = Lamb(parameter_groups, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        if self.optimizer_verbose:
            optimizer = VerboseOptimizer(optimizer)

        # Initialize scheduler
        if self.scheduler is None:
            scheduler = []
        elif self.scheduler == 'cycling':
            scheduler = [{'scheduler': torch.optim.lr_scheduler.CyclicLR(
                            optimizer,
                            base_lr        = 1e-4,
                            max_lr         = 5e-3,
                            step_size_up   = 10,
                            cycle_momentum = False,
                            verbose        = True),
                        'interval': 'epoch',
                        'monitor' : 'val_loss'
                        }]
        elif self.scheduler == 'plateau':
            scheduler = [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            patience = self.hparams['patience'],
                            factor   = self.hparams['factor'],
                            mode     = 'min',
                            verbose  = True),
                        'interval': 'epoch',
                        'monitor' : 'train_loss'
                        }]
        else:
            raise ValueError(f'Unknown scheduler: {self.scheduler}')

        return [optimizer], scheduler

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_index):
        """Train model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        y_hat   = self(X_batch)
        loss    = self.loss(y_hat, y_batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        """Validate model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        y_hat   = self(X_batch)
        loss    = self.loss(y_hat, y_batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def test_step(self, batch, batch_index):
        """Test model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        y_hat   = self(X_batch)
        # Return predictions
        return {'y': y_batch[:,0].detach().cpu(), 'y_hat': y_hat[:,0].detach().cpu()}

    def test_epoch_end(self, test_step_outputs):
        # Collect predictions from individual batches
        y     = torch.tensor([])
        y_hat = torch.tensor([])
        for output in test_step_outputs:
            y     = torch.cat((y    , output['y']))
            y_hat = torch.cat((y_hat, output['y_hat']))
        # Save predictions for evaluation
        self.test_y     = y
        self.test_y_hat = y_hat