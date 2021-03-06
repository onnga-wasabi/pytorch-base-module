import json
from typing import Dict, Union
from tqdm import tqdm
import dataclasses

import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import Config


@dataclasses.dataclass
class State:
    epoch: int
    iteration: int
    epoch_pbar: tqdm
    iteration_pbar: tqdm
    best: float


class BaseTrainer(object):

    def __init__(
        self,
        config: Config,
        network: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        log_dir: str,
        val_data_loader: torch.utils.data.DataLoader = None,
        wandb_project_name: str = None,
    ):
        self.config = config
        self.network = network
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.results = {'train': []}

        if val_data_loader:
            self.val_data_loader = val_data_loader
            self.val_data_iter = iter(self.val_data_loader)
            self.results['val'] = []

        self.device_setup()
        self.wandb_setup(wandb_project_name)

        self.state = State(
            epoch=0,
            iteration=0,
            epoch_pbar=tqdm(total=self.config.experiment.epoch, leave=False, ncols=100),
            iteration_pbar=tqdm(total=len(self.data_iter), leave=False, ncols=150),
            best=np.inf,
        )

        self.optimizer_setup()
        self.extra_setup()

    def device_setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

        if torch.cuda.device_count() > 1:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.network = nn.DataParallel(self.network)

    def wandb_setup(self, wandb_project_name):
        if wandb_project_name:
            self.wandb = wandb
            self.wandb.init(
                name=self.config.name + "-" + self.wandb.run.id,
                project=wandb_project_name,
                config=dataclasses.asdict(self.config),
            )
            network = self.network.module if isinstance(self.network, nn.DataParallel) else self.network
            self.wandb.watch(network, log='all')
        else:
            self.wandb = None

    def extra_setup(self):
        pass

    def save(self):
        if isinstance(self.network, nn.DataParallel):
            self.network = self.network.module
        self.network = self.network.to('cpu')
        torch.save(self.network.state_dict(), f'{self.log_dir}/model.pt')

    def fit(self):
        while self.state.epoch < self.config.experiment.epoch:
            self.update()

            # evalation
            if self.val_data_loader:
                self.network.eval()
                self.evaluate()
                self.network.train()

            self.new_epoch()

        self.fitting_end()

    def update(self):
        results = []
        while True:
            try:
                self.optimizer.zero_grad()
                batch = next(self.data_iter)
                computed = self.compute(batch)
                computed['loss'].backward()
                self.optimizer.step()

                self.state.iteration += 1
                self.state.iteration_pbar.update(1)
                self.state.iteration_pbar.set_postfix(computed['log'])
                self.iteration_end(computed)
                results.append(computed)

            except StopIteration:
                break
        self.update_end(results)

    def iteration_end(self, computed):
        for k, v in computed['log'].items():
            self.writer.add_scalar(f'Train/{k}', v, self.state.iteration)
            if self.wandb:
                self.wandb.log({f'Train/{k}': v})

        computed['log']['iteration'] = self.state.iteration
        computed['log']['epoch'] = self.state.epoch
        self.results['train'].append(computed['log'])

    def update_end(self, results):
        pass

    def evaluate(self):
        results = []
        while True:
            try:
                batch = next(self.val_data_iter)
                computed = self.evaluate_func(batch)
                results.append(computed)
            except StopIteration:
                break
        self.evaluate_end(results)

    @torch.no_grad()
    def evaluate_func(self, batch):
        return self.compute(batch)

    def evaluate_end(self, results, condition_key: str = 'Loss'):
        logs = {}
        for k in results[0]['log'].keys():
            metric = np.mean([r['log'][k] for r in results])
            logs[k] = metric
            self.writer.add_scalar(f'Validation/{k}', metric, self.state.epoch)
            if self.wandb:
                self.wandb.log({f'Validation/{k}': metric})

        logs['epoch'] = self.state.epoch
        self.results['val'].append(logs)
        self.save_weights(condition_key)

    def new_epoch(self):
        self.epoch_end()
        self.state.epoch += 1
        self.state.epoch_pbar.update(1)
        self.state.iteration_pbar.close()
        self.state.iteration_pbar = tqdm(total=len(self.data_iter), leave=False, ncols=150)

        del(self.data_iter)
        self.data_iter = iter(self.data_loader)

        with open(f'{self.log_dir}/log.json', 'w') as wf:
            json.dump(self.results['train'], wf, indent=2)

        if self.val_data_iter:
            del(self.val_data_iter)
            self.val_data_iter = iter(self.val_data_loader)
            with open(f'{self.log_dir}/val_log.json', 'w') as wf:
                json.dump(self.results['val'], wf, indent=2)

    def epoch_end(self):
        """
        画像とかをhogehogeするなかここかな
        """
        pass

    def fitting_end(self):
        self.writer.close()
        self.state.epoch_pbar.close()
        self.save()

    @torch.no_grad()
    def save_weights(self, condition_key: str):
        if self.results['val'][-1][condition_key] < self.state.best:
            if isinstance(self.network, nn.DataParallel):
                self.network = self.network.module
            self.network = self.network.to('cpu')
            torch.save(self.network.state_dict(), f'{self.log_dir}/model_best.pt')
            self.device_setup()
            self.state.best = self.results['val'][-1][condition_key]
        else:
            pass

    def compute(self, batch) -> Dict:
        raise NotImplementedError

    def optimizer_setup(self):
        self.optimizer: Union[torch.optim.Adam, None] = None
        raise NotImplementedError


class BaseGANTrainer(BaseTrainer):

    def __init__(
        self,
        config: Config,
        network_D: nn.Module,
        network_G: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        log_dir: str,
        val_data_loader: torch.utils.data.DataLoader = None,
        wandb_project_name: str = None,
    ):
        self.config = config
        self.network_D = network_D
        self.network_G = network_G
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.results = {'train': []}

        if val_data_loader:
            self.val_data_loader = val_data_loader
            self.val_data_iter = iter(self.val_data_loader)
            self.results['val'] = []

        self.state = State(
            epoch=0,
            iteration=0,
            epoch_pbar=tqdm(total=self.config.experiment.epoch, leave=False, ncols=100),
            iteration_pbar=tqdm(total=len(self.data_loader), leave=False, ncols=150),
            best=np.inf,
        )

        self.device_setup()
        self.wandb_setup(wandb_project_name)
        self.optimizer_setup()
        self.extra_setup()

    def device_setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network_D.to(self.device)
        self.network_G.to(self.device)

        if torch.cuda.device_count() > 1:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.network_D = nn.DataParallel(self.network_D)
            self.network_G = nn.DataParallel(self.network_G)

    def fit(self):
        while self.state.epoch < self.config.experiment.epoch:
            self.update()

            # evalation
            if self.val_data_loader:
                self.network_D.eval()
                self.network_G.eval()
                self.evaluate()
                self.network_D.train()
                self.network_G.train()

            self.new_epoch()

        self.fitting_end()

    def compute(self, batch):
        pass

    def compute_D(self, batch) -> Dict:
        raise NotImplementedError

    def compute_G(self, batch) -> Dict:
        raise NotImplementedError

    def optimizer_setup(self):
        self.optimizer_D: Union[torch.optim.Adam, None] = None
        self.optimizer_G: Union[torch.optim.Adam, None] = None
        raise NotImplementedError

    def update(self):
        results = []
        while True:
            try:
                batch = next(self.data_iter)
                computed = {'log': {}}

                # Discriminator
                self.optimizer_D.zero_grad()
                computed_D = self.compute_D(batch)
                computed_D['loss'].backward()
                self.optimizer_D.step()
                for k, v in computed_D['log'].items():
                    key = 'D_' + k
                    computed['log'][key] = v

                # Generator
                self.optimizer_G.zero_grad()
                computed_G = self.compute_G(batch)
                computed_G['loss'].backward()
                self.optimizer_G.step()
                for k, v in computed_G['log'].items():
                    key = 'G_' + k
                    computed['log'][key] = v

                self.state.iteration += 1
                self.state.iteration_pbar.update(1)
                self.state.iteration_pbar.set_postfix(computed['log'])
                self.iteration_end(computed)
                results.append(computed)

            except StopIteration:
                break
        self.update_end(results)

    @torch.no_grad()
    def evaluate(self):
        results = []
        while True:
            try:
                computed = {'log': {}}
                batch = next(self.val_data_iter)
                computed_D = self.compute_D(batch)
                for k, v in computed_D['log'].items():
                    key = 'D_' + k
                    computed['log'][key] = v

                computed_G = self.compute_G(batch)
                for k, v in computed_G['log'].items():
                    key = 'G_' + k
                    computed['log'][key] = v

                results.append(computed)
            except StopIteration:
                break
        self.evaluate_end(results, condition_key='G_Loss')

    @torch.no_grad()
    def save_weights(self, condition_key: str):
        if self.results['val'][-1][condition_key] < self.state.best:
            if isinstance(self.network_D, nn.DataParallel):
                self.network_D = self.network_D.module
                self.network_G = self.network_G.module
            self.network_D = self.network_D.to('cpu')
            self.network_G = self.network_G.to('cpu')

            torch.save(self.network_D.state_dict(), f'{self.log_dir}/model_D_best.pt')
            torch.save(self.network_G.state_dict(), f'{self.log_dir}/model_G_best.pt')

            self.device_setup()
            self.state.best = self.results['val'][-1][condition_key]
        else:
            pass

    def save(self):
        if isinstance(self.network_D, nn.DataParallel):
            self.network_D = self.network_D.module
            self.network_G = self.network_G.module
        self.network_D = self.network_D.to('cpu')
        self.network_G = self.network_G.to('cpu')
        torch.save(self.network_D.state_dict(), f'{self.log_dir}/model_G.pt')
        torch.save(self.network_G.state_dict(), f'{self.log_dir}/model_D.pt')
