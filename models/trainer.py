import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import DataLoader

from data import VCDataset, collate_fn
from .model import TTSModel
from .loss import mle_loss
from .lr_scheduler import NoamLR
from utils import seed_everything, Tracker


class Trainer:
    def __init__(self, config_path):
        self.config_path = config_path

    def run(self):
        config = OmegaConf.load(self.config_path)

        accelerator = Accelerator(fp16=config.train.fp16)

        seed_everything(config.seed)

        output_dir = Path(config.model_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        OmegaConf.save(config, output_dir / 'config.yaml')

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=f'{str(output_dir)}/logs')
        else:
            writer = None

        train_data, valid_data = self.prepare_data(config.data)
        train_dataset = VCDataset(train_data)
        valid_dataset = VCDataset(valid_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.train.batch_size,
            num_workers=8,
            collate_fn=collate_fn
        )

        model = TTSModel(config.model)
        optimizer = optim.AdamW(model.parameters(), eps=1e-9, **config.optimizer)

        epochs = self.load(config, model, optimizer)

        model, optimizer, train_loader, valid_loader = accelerator.prepare(
            model, optimizer, train_loader, valid_loader
        )
        scheduler = NoamLR(optimizer, d_model=config.model.encoder.channels, last_epoch=epochs * len(train_loader) - 1)

        for epoch in range(epochs, config.train.num_epochs):
            self.train_step(epoch, model, optimizer, scheduler, train_loader, writer, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                self.valid_step(epoch, model, valid_loader, writer)
                if (epoch + 1) % config.train.save_interval == 0:
                    self.save(
                        output_dir / 'latest.ckpt',
                        epoch,
                        (epoch+1)*len(train_loader),
                        accelerator.unwrap_model(model),
                        optimizer
                    )
        if accelerator.is_main_process:
            writer.close()

    def train_step(self, epoch, model, optimizer, scheduler, loader, writer, accelerator):
        model.train()
        tracker = Tracker()
        bar = tqdm(desc=f'Epoch: {epoch + 1}', total=len(loader), disable=not accelerator.is_main_process)
        for i, batch in enumerate(loader):
            loss = self._handle_batch(batch, model, tracker)
            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            bar.update()
            bar.set_postfix_str(f'Loss: {loss:.6f}')
        bar.set_postfix_str(f'Mean Loss: {tracker.loss.mean():.6f}')
        if accelerator.is_main_process:
            self.write_losses(epoch, writer, tracker, mode='train')
        bar.close()

    def valid_step(self, epoch, model, loader, writer):
        model.eval()
        tracker = Tracker()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                _ = self._handle_batch(batch, model, tracker)
        self.write_losses(epoch, writer, tracker, mode='valid')

    def _handle_batch(self, batch, model, tracker):
        (
            mel,
            phoneme,
            a1,
            f2,
            pitch,
            energy,
            duration,
            x_length,
            y_length
        ) = batch
        x, (z, log_df_dz), dur_pred, (x_mask, y_mask) = model(
            phoneme, a1, f2, x_length, mel, y_length, duration
        )
        loss_mle = mle_loss(z, x, log_df_dz, y_length)
        tgt_dur = torch.log(duration + 1e-4) * x_mask
        loss_duration = F.mse_loss(dur_pred, tgt_dur.to(x.dtype))
        print(loss_mle.dtype, loss_duration.dtype)
        loss = loss_mle + loss_duration
        tracker.update(
            loss=loss.item(),
            mle=loss_mle.item(),
            duration=loss_duration.item()
        )
        return loss

    def prepare_data(self, config):
        data_dir = Path(config.data_dir)
        assert data_dir.exists()

        fns = list(sorted(list(data_dir.glob('*.pt'))))
        valid_size = 100
        valid = fns[:valid_size]
        train = fns[valid_size:]
        return train, valid

    def load(self, config, model, optimizer):
        if config.resume_checkpoint:
            checkpoint = torch.load(f'{config.model_dir}/latest.ckpt')
            epochs = checkpoint['epoch']
            iteration = checkpoint['iteration']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Loaded {iteration}iter model and optimizer.')
            return epochs + 1
        else:
            return 0

    def save(self, save_path, epoch, iteration, model, optimizer):
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)

    def write_losses(self, epoch, writer, tracker, mode='train'):
        for k, v in tracker.items():
            writer.add_scalar(f'{mode}/{k}', v.mean(), epoch)
