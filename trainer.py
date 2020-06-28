import json
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

        if wandb_project_name:
            self.wandb = wandb
            self.wandb.init(
                name=config.name,
                project=wandb_project_name,
                config=dataclasses.asdict(config),
            )
        else:
            self.wandb = None

        self.results = {'train': []}

        if val_data_loader:
            self.val_data_loader = val_data_loader
            self.val_data_iter = iter(self.val_data_loader)
            self.results['val'] = []

        self.state = State(
            epoch=0,
            iteration=0,
            epoch_pbar=tqdm(total=self.config.experiment.epoch, leave=False, ncols=100),
            iteration_pbar=tqdm(total=len(self.data_iter), leave=False, ncols=150),
        )

        self.device_setup()
        self.optimizer_setup()
        self.extra_setup()

    def device_setup(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)

        if torch.cuda.device_count() > 1:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            self.network = nn.DataParallel(self.network)
            if self.wandb:
                self.wandb.watch(self.network.module, log='all')

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

    def evaluate_end(self, results):
        logs = {}
        for k in results[0]['log'].keys():
            metric = np.mean([r['log'][k] for r in results])
            logs[k] = metric
            self.writer.add_scalar(f'Validation/{k}', metric, self.state.epoch)
            if self.wandb:
                self.wandb.log({f'Validation/{k}': metric})

        logs['epoch'] = self.state.epoch
        self.results['val'].append(logs)

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

    def compute(self, batch):
        raise NotImplementedError

    def optimizer_setup(self):
        raise NotImplementedError
