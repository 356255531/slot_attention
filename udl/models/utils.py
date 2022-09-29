import torch


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return torch.nn.ReLU()
    elif act_fn == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif act_fn == 'elu':
        return torch.nn.ELU()
    elif act_fn == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act_fn == 'softplus':
        return torch.nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)