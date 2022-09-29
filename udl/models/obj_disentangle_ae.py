import numpy as np
import torch
import pytorch_lightning as pl

from udl.models.utils import get_act_fn, to_one_hot


class EncoderCNNSmall(torch.nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNSmall, self).__init__()
        self.cnn1 = torch.nn.Conv2d(
            input_dim, hidden_dim, (10, 10), stride=10)
        self.cnn2 = torch.nn.Conv2d(hidden_dim, num_objects, (1, 1), stride=1)
        self.ln1 = torch.nn.BatchNorm2d(hidden_dim)
        self.act1 = get_act_fn(act_fn_hid)
        self.act2 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        return self.act2(self.cnn2(h))


class EncoderCNNMedium(torch.nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='leaky_relu'):
        super(EncoderCNNMedium, self).__init__()

        self.cnn1 = torch.nn.Conv2d(
            input_dim, hidden_dim, (9, 9), padding=4)
        self.act1 = get_act_fn(act_fn_hid)
        self.ln1 = torch.nn.BatchNorm2d(hidden_dim)

        self.cnn2 = torch.nn.Conv2d(
            hidden_dim, num_objects, (5, 5), stride=5)
        self.act2 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.cnn2(h))
        return h


class EncoderCNNLarge(torch.nn.Module):
    """CNN encoder, maps observation to obj-specific feature maps."""

    def __init__(self, input_dim, hidden_dim, num_objects, act_fn='sigmoid',
                 act_fn_hid='relu'):
        super(EncoderCNNLarge, self).__init__()

        self.cnn1 = torch.nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.act1 = get_act_fn(act_fn_hid)
        self.ln1 = torch.nn.BatchNorm2d(hidden_dim)

        self.cnn2 = torch.nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act2 = get_act_fn(act_fn_hid)
        self.ln2 = torch.nn.BatchNorm2d(hidden_dim)

        self.cnn3 = torch.nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.act3 = get_act_fn(act_fn_hid)
        self.ln3 = torch.nn.BatchNorm2d(hidden_dim)

        self.cnn4 = torch.nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)
        self.act4 = get_act_fn(act_fn)

    def forward(self, obs):
        h = self.act1(self.ln1(self.cnn1(obs)))
        h = self.act2(self.ln2(self.cnn2(h)))
        h = self.act3(self.ln3(self.cnn3(h)))
        return self.act4(self.cnn4(h))


class EncoderMLP(torch.nn.Module):
    """MLP encoder, maps observation to latent state."""

    def __init__(self, input_dim, output_dim, hidden_dim, num_objects,
                 act_fn='relu'):
        super(EncoderMLP, self).__init__()

        self.num_objects = num_objects
        self.input_dim = input_dim

        self.fc1 = torch.nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

        self.ln = torch.nn.LayerNorm(hidden_dim)

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)

    def forward(self, ins):
        h_flat = ins.view(-1, self.num_objects, self.input_dim)
        h = self.act1(self.fc1(h_flat))
        h = self.act2(self.ln(self.fc2(h)))
        return self.fc3(h)


class DecoderMLP(torch.nn.Module):
    """MLP decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderMLP, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim + num_objects, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, np.prod(output_size))

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.output_size = output_size

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)

    def forward(self, ins):
        obj_ids = torch.arange(self.num_objects)
        obj_ids = to_one_hot(obj_ids, self.num_objects).unsqueeze(0)
        obj_ids = obj_ids.repeat((ins.size(0), 1, 1)).to(ins.get_device())

        h = torch.cat((ins, obj_ids), -1)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.fc3(h).sum(1)
        return h.view(-1, self.output_size[0], self.output_size[1],
                      self.output_size[2])


class DecoderCNNSmall(torch.nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNSmall, self).__init__()

        width, height = output_size[1] // 10, output_size[2] // 10

        output_dim = width * height

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)

        self.deconv1 = torch.nn.ConvTranspose2d(1, hidden_dim,
                                          kernel_size=1, stride=1)
        self.deconv2 = torch.nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=10, stride=10)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)

    def forward(self, ins):
        ins_flat = ins.view(-1, self.input_dim)
        h = self.act1(self.fc1(ins_flat))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, 1, self.map_size[1], self.map_size[2])
        h = self.act3(self.deconv1(h_conv))
        h = self.deconv2(h).view((-1, self.num_objects,) + h.shape[-2:])
        import pdb
        pdb.set_trace()
        return self.deconv2(h)


class DecoderCNNMedium(torch.nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNMedium, self).__init__()

        width, height = output_size[1] // 5, output_size[2] // 5

        output_dim = width * height

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)

        self.deconv1 = torch.nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=5, stride=5)
        self.deconv2 = torch.nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=9, padding=4)

        self.ln1 = torch.nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        return self.deconv2(h)


class DecoderCNNLarge(torch.nn.Module):
    """CNN decoder, maps latent state to image."""

    def __init__(self, input_dim, hidden_dim, num_objects, output_size,
                 act_fn='relu'):
        super(DecoderCNNLarge, self).__init__()

        width, height = output_size[1], output_size[2]

        output_dim = width * height

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)

        self.deconv1 = torch.nn.ConvTranspose2d(num_objects, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(hidden_dim, hidden_dim,
                                          kernel_size=3, padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(hidden_dim, output_size[0],
                                          kernel_size=3, padding=1)

        self.ln1 = torch.nn.BatchNorm2d(hidden_dim)
        self.ln2 = torch.nn.BatchNorm2d(hidden_dim)
        self.ln3 = torch.nn.BatchNorm2d(hidden_dim)

        self.input_dim = input_dim
        self.num_objects = num_objects
        self.map_size = output_size[0], width, height

        self.act1 = get_act_fn(act_fn)
        self.act2 = get_act_fn(act_fn)
        self.act3 = get_act_fn(act_fn)
        self.act4 = get_act_fn(act_fn)
        self.act5 = get_act_fn(act_fn)

    def forward(self, ins):
        h = self.act1(self.fc1(ins))
        h = self.act2(self.ln(self.fc2(h)))
        h = self.fc3(h)

        h_conv = h.view(-1, self.num_objects, self.map_size[1],
                        self.map_size[2])
        h = self.act3(self.ln1(self.deconv1(h_conv)))
        h = self.act4(self.ln1(self.deconv2(h)))
        h = self.act5(self.ln1(self.deconv3(h)))
        return self.deconv4(h)


class ObjDisentangleAE(pl.LightningModule):
    def __init__(self, exp_params):
        super().__init__()

        self.exp_params = exp_params

        width_height = np.array(exp_params.data.img_dim[1:])
        if exp_params.model.encoder == 'small':
            self.feat_extractor = EncoderCNNSmall(
                input_dim=exp_params.model.input_channels,
                hidden_dim=exp_params.model.hidden_dim // 16,
                num_objects=exp_params.data.num_objects)
            # CNN image size changes
            width_height //= 10
        elif exp_params.model.encoder == 'medium':
            self.feat_extractor = EncoderCNNMedium(
                input_dim=exp_params.model.input_channels,
                hidden_dim=exp_params.model.hidden_dim // 16,
                num_objects=exp_params.data.num_objects)
            # CNN image size changes
            width_height //= 5
        elif exp_params.model.encoder == 'large':
            self.feat_extractor = EncoderCNNLarge(
                input_dim=exp_params.model.input_channels,
                hidden_dim=exp_params.model.hidden_dim // 16,
                num_objects=exp_params.data.num_objects)
        else:
            raise ValueError(f"Encoder {exp_params.model.encoder} not exists!")

        self.mlp_encoder = EncoderMLP(
            input_dim=np.prod(width_height),
            hidden_dim=exp_params.model.hidden_dim,
            output_dim=exp_params.model.embedding_dim,
            num_objects=exp_params.data.num_objects)

        if exp_params.model.encoder == 'large':
            self.decoder = DecoderCNNLarge(
                input_dim=exp_params.model.embedding_dim,
                num_objects=exp_params.data.num_objects,
                hidden_dim=exp_params.model.hidden_dim // 16,
                output_size=exp_params.data.img_dim)
        elif exp_params.model.encoder == 'medium':
            self.decoder = DecoderCNNMedium(
                input_dim=exp_params.model.embedding_dim,
                num_objects=exp_params.data.num_objects,
                hidden_dim=exp_params.model.hidden_dim // 16,
                output_size=exp_params.data.img_dim)
        elif exp_params.model.encoder == 'small':
            self.decoder = DecoderCNNSmall(
                input_dim=exp_params.model.embedding_dim,
                num_objects=exp_params.data.num_objects,
                hidden_dim=exp_params.model.hidden_dim // 16,
                output_size=exp_params.data.img_dim)

    def forward(self, x):
        return self.decoder(self.mlp_encoder(self.feat_extractor(x))).sum(dim=1)

    def training_step(self, x, batch_idx):
        return torch.nn.functional.mse_loss(self(x), x)

    def validation_step(self, x, batch_idx):
        return torch.nn.functional.mse_loss(self(x), x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.exp_params.train.lr)