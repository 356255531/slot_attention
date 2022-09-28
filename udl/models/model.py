from typing import Tuple

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import utils as vutils

import pytorch_lightning as pl

from udl.utils import Tensor
from udl.utils import assert_shape
from udl.utils import build_grid
from udl.utils import conv_transpose_out_shape
from udl.utils import group_transformation
from udl.utils import to_rgb_from_tensor


class SlotAttention(torch.nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = torch.nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = torch.nn.LayerNorm(self.slot_size)
        self.norm_mlp = torch.nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = torch.nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = torch.nn.Linear(self.slot_size // 2, self.slot_size, bias=False)
        self.project_v = torch.nn.Linear(self.slot_size // 2, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = torch.nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.slot_size, self.mlp_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            torch.nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=torch.nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            torch.nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=torch.nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots


class SlotAttentionModel(pl.LightningModule):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations,
        train_dataloader,
        val_dataloader,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution=(6, 6),
        empty_cache=False,
        params=None
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.params = params

        self.out_features = self.hidden_dims[-1]

        self.encoder = C8SteerableCNN()
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, resolution)
        self.encoder_out_layer = torch.nn.Sequential(
            torch.nn.Linear(self.out_features, self.out_features),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    torch.nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.ConvTranspose2d(self.out_features, 3, kernel_size=3, stride=1, padding=1, output_padding=0,),
            )
        )

        assert_shape(resolution, (out_size, out_size), message="")

        self.decoder = torch.nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size * 2,
            mlp_hidden_size=128,
        )

        self.t_params_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.slot_size, 2 * self.slot_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.slot_size * 2, 6),
        )

    def forward(self, x):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [batch_size, filter_size, height, width]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [batch_size, filter_size, height*width]
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [batch_size, height*width, filter_size]

        slots = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size * 2))
        # `slots` has shape: [batch_size, num_slots, slot_size].
        slots, params = slots[:, :, :self.slot_size], self.t_params_mlp(slots[:, :, self.slot_size:])
        batch_size, num_slots, slot_size = slots.shape

        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(out.size(), (batch_size * num_slots, num_channels, height, width))

        recons = out.reshape(batch_size, num_slots, num_channels, height, width)
        recon_combined = torch.sum(recons, dim=1)
        transformed_recons = group_transformation(recons, params)
        transformed_recons_combined = torch.sum(transformed_recons, dim=1)
        return recon_combined, recons, transformed_recons_combined, transformed_recons, slots

    def loss_function(self, input):
        recon_combined, recons, transformed_recons_combined, transformed_recons, slots = self.forward(input)

        reconstruction_loss = F.mse_loss(transformed_recons_combined, input)
        return {
            "loss": reconstruction_loss
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        warmup_steps_pct = self.params.warmup_steps_pct
        decay_steps_pct = self.params.decay_steps_pct
        total_steps = self.params.max_epochs * len(self.train_dataloader)

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step < total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.params.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.val_dataloader
        perm = torch.randperm(self.params.batch_size)
        idx = perm[: self.params.n_samples]
        batch = next(iter(dl))[idx]
        if len(self.params.gpus) > 0:
            batch = batch.to(self.device)
        recon_combined, recons, transformed_recons_combined, transformed_recons, slots = self.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid, output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    batch.unsqueeze(1),  # original images
                    transformed_recons_combined.unsqueeze(1),  # reconstructions
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons,  # each slot
                    transformed_recons
                ],
                dim=1,
            )
        )

        batch_size, num_slots, C, H, W = recons.shape
        out = out.permute(1, 0, 2, 3, 4)
        images = vutils.make_grid(
            out.reshape(out.shape[0] * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))


class SoftPositionEmbed(torch.nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = torch.nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
