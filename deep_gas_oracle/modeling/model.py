from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .dataset import TimeSeriesDataset
from ..preprocessing.norm import Normalizer


class GruMultiStep(nn.Module):

    def __init__(self,
                 features: list,
                 targets: list,
                 input_length: int = 200,
                 output_length: int = 5,
                 hidden_size: int = 32,
                 num_layers: int = 1,
                 learning_rate: float = 0.001,
                 linear_gain: bool = True,
                 norm_clip: float = 1.0,
                 smooth_fraction: float = 0.6,
                 device: str = "cuda:0",
                 logs_dir: str = "gas_predictor/runs"):

        # Parameters
        super().__init__()
        self.features = features
        self.targets = targets
        self.input_size = len(features)
        self.input_length = input_length
        self.output_size = len(targets)
        self.output_length = output_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.smooth_fraction = smooth_fraction
        self.device = torch.device(device)
        # Model definition
        self.gru = nn.GRU(self.input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.linear_head = nn.Linear(hidden_size, self.output_size * output_length)
        # Loss and optimizer
        self.loss_func = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.norm_clip = norm_clip
        self.linear_gain = linear_gain
        if linear_gain:
            self.linear_space = torch.linspace(0.0, 1.0, input_length,
                                               device=device,
                                               dtype=torch.float32)
        # Move the weights to the selected hardware
        self.to(self.device)
        # Tensorboard writer
        if not Path(logs_dir).exists():
            Path(logs_dir).mkdir(parents=True)
        count = len([p.stem for p in Path(logs_dir).iterdir()]) + 1
        self.model_path = Path(logs_dir) / f"exp_{count}"

        print("Parameters (param name -> param count):")
        for pname, pparams in self.named_parameters():
            pcount = np.prod(pparams.size())
            print(f"\t{pname} -> {pcount}")

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        param_count = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Total param count: {param_count}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gru_out, hidden_state = self.gru(x)
        return self.linear_head(gru_out).reshape(-1,  # batch size
                                                 x.shape[1],  # input length
                                                 self.output_length,  # prediction length
                                                 self.output_size)  # dim targets

    def _loss(self,
              output: torch.Tensor,
              target: torch.Tensor) -> torch.Tensor:

        if self.linear_gain:
            loss = (output - target) ** 2
            for i in range(self.linear_space.shape[0]):
                loss[:, i, :, :] = loss[:, i, :, :] * self.linear_space[i]
        else:
            loss = (output - target) ** 2
        loss = torch.mean(loss)
        return loss

    def fit(self,
            train_df: pd.DataFrame,
            val_df: pd.DataFrame,
            batch_size: int = 32,
            epochs: int = 100):

        self.writer = SummaryWriter(str(self.model_path))
        # Data initalisation
        train_ds = TimeSeriesDataset(train_df,
                                     self.features,
                                     self.targets,
                                     self.input_length,
                                     self.output_length,
                                     self.smooth_fraction)

        val_ds = TimeSeriesDataset(val_df,
                                   self.features,
                                   self.targets,
                                   self.input_length,
                                   self.output_length,
                                   self.smooth_fraction)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 20,
            shuffle=False,
            drop_last=False,
        )

        estimated_steps = int(len(train_ds) / batch_size)
        print("Train Dataset Len: {}".format(len(train_ds)))
        print(f"Estimated steps train: {estimated_steps}")

        # Training loop
        best_loss_value = float("inf")
        for epoch in range(1, epochs + 1):
            self.train()
            # Loss init
            train_loss = 0
            for train_batch in train_loader:
                x_train = train_batch["x"]
                y_train = train_batch["y"]
                # Move the inputs/outputs tensors to the selected hardware
                x_train = x_train.to(dtype=torch.float32, device=self.device)
                y_train = y_train.to(dtype=torch.float32, device=self.device)
                # Forward pass
                preds = self(x_train)
                # Backpropagate the errors through the network
                self.optim.zero_grad()
                loss = self._loss(preds, y_train)
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.parameters(), self.norm_clip
                )

                self.optim.step()
                # record loss
                train_loss += loss.item()

            # Check the performance on the valiation data
            self.eval()
            # Init validation loss
            val_loss = 0
            val_loss_per_step = {i: 0 for i in range(self.output_length)}
            for val_batch in val_loader:
                x_val = val_batch["x"]
                y_val = val_batch["y"]
                # Move the inputs/outputs tensors to the selected hardware
                x_val = x_val.to(dtype=torch.float32, device=self.device)
                y_val = y_val.to(dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    preds = self(x_val)
                    loss = self._loss(preds, y_val)
                    val_loss += loss.item()

                    for i in range(self.output_length):
                        loss = self._loss(preds[:, :, i:i + 1, :], y_val[:, :, i:i + 1, :])
                        val_loss_per_step[i] += loss.item()

            self.writer.add_scalars('loss', {'train': train_loss / len(train_loader),
                                             'val': val_loss / len(val_loader)}, epoch)
            for i in range(self.output_length):
                self.writer.add_scalars(f'loss_timestep_{i+1}',
                                        {'val': val_loss_per_step[i] / len(val_loader)}, epoch)
            print(f"EPOCH {epoch} completed:")
            print(f"  train loss: {train_loss / len(train_loader)}")
            print(f"  val loss: {val_loss / len(val_loader)}\n")
            if not (self.model_path / "weights").exists():
                (self.model_path / "weights").mkdir(exist_ok=True)
            torch.save(self.state_dict(), self.model_path / "weights" / f"epoch_{epoch}.pth")
            if val_loss < best_loss_value:
                torch.save(self.state_dict(), self.model_path / "weights" / f"epoch_best.pth")

    def predict(self,
                df: pd.DataFrame,
                scaler: Normalizer = None,
                normalize: bool = False,
                denormalize: bool = False,
                use_ground_truth: bool = True,
                batch_size=100):

        self.eval()

        if (normalize or denormalize) and scaler is None:
            raise RuntimeError(f"You cannot normalize or denormalize if you don't pass a scaler in the parameters")

        if normalize:
            df = scaler.transform(df)

        if use_ground_truth:
            targets = self.targets
        else:
            targets = []

        ds = TimeSeriesDataset(df,
                               self.features,
                               targets,
                               self.input_length,
                               self.output_length,
                               self.smooth_fraction)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        preds_cols = []
        for i in range(self.output_length):  # for each prediction step
            preds_cols += [c + f"_{i}" for c in self.targets]

        df_predictions = None
        df_ground_truth = None
        for batch in loader:
            x = batch["x"]
            x = x.to(dtype=torch.float32, device=self.device)
            # Formatting dates
            t = batch["t"].numpy()
            ts = pd.to_datetime(t, unit="us")

            # Make prediction
            with torch.no_grad():
                predictions = self(x)
            predictions = predictions[:, -1].detach().cpu().numpy()  # last predictions of each window only
            predictions = predictions.reshape(-1, predictions.shape[1] * predictions.shape[2])

            # Get ground truth
            if use_ground_truth:
                y = batch["y"][:, -1].numpy()  # last predictions of each window only
                y = y.reshape(-1, y.shape[1] * y.shape[2])

            if df_predictions is None:
                df_predictions = pd.DataFrame(data=predictions, columns=preds_cols, index=ts)
                if use_ground_truth:
                    df_ground_truth = pd.DataFrame(data=y, columns=preds_cols, index=ts)
            else:
                df_pred = pd.DataFrame(data=predictions, columns=preds_cols, index=ts)
                df_predictions = pd.concat([df_predictions, df_pred], axis=0)
                if use_ground_truth:
                    df_y = pd.DataFrame(data=y, columns=preds_cols, index=ts)
                    df_ground_truth = pd.concat([df_ground_truth, df_y], axis=0)

        if not denormalize:
            if use_ground_truth:
                return df_predictions, df_ground_truth
            else:
                return df_predictions
        else:
            if use_ground_truth:
                return scaler.invert(df_predictions), scaler.invert(df_ground_truth)
            else:
                return scaler.invert(df_predictions)
