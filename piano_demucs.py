from demucs.htdemucs import HTDemucs
from demucs.solver import Solver
from data import PianoDataset
import torch
from munch import DefaultMunch
import logging

logging.basicConfig(level=logging.INFO)

handler = logging.StreamHandler()
logger = logging.getLogger(__name__)
logger.addHandler(handler)

misc = {
    'num_workers': 10,
    'num_prints': 4,
    'show': False,
    'verbose': False
}

optim_params = { 'lr': 3e-4,
    'momentum': 0.9,
    'beta2': 0.999,
    'loss': 'l1',
    'optim': 'adamw',
    'weight_decay': 0,
    'clip_grad': 2.0
}

ht_demucs_params = {
    'channels': 48,
    'channels_time': None,
    'growth': 2,
    'nfft': 4096,
    'wiener_iters': 0,
    'end_iters': 0,
    'wiener_residual': False,
    'cac': True,
    'depth': 4,
    'rewrite': True,
    'multi_freqs': [],
    'multi_freqs_depth': 3,
    'freq_emb': 0.2,
    'emb_scale': 10,
    'emb_smooth': True,
    'kernel_size': 8,
    'stride': 4,
    'time_stride': 2,
    'context': 1,
    'context_enc': 0,
    'norm_starts': 4,
    'norm_groups': 4,
    'dconv_mode': 1,
    'dconv_depth': 2,
    'dconv_comp': 8,
    'dconv_init': 1e-3,
    'bottom_channels': 0,
    't_layers': 5,
    't_hidden_scale': 4.0,
    't_heads': 8,
    't_dropout': 0.0,
    't_layer_scale': True,
    't_gelu': True,
    't_emb': 'sin',
    't_max_positions': 10000,
    't_max_period': 10000.0,
    't_weight_pos_embed': 1.0,
    't_cape_mean_normalize': True,
    't_cape_augment': True,
    't_cape_glob_loc_scale': [5000.0, 1.0, 1.4],
    't_sin_random_shift': 0,
    't_norm_in': True,
    't_norm_in_group': False,
    't_group_norm': False,
    't_norm_first': True,
    't_norm_out': True,
    't_weight_decay': 0.0,
    't_lr': None,
    't_sparse_self_attn': False,
    't_sparse_cross_attn': False,
    't_mask_type': 'diag',
    't_mask_random_seed': 42,
    't_sparse_attn_window': 400,
    't_global_window': 100,
    't_sparsity': 0.95,
    't_auto_sparsity': False,
    't_cross_first': False,
    'rescale': 0.1
}

test = {
    'save': False,
    'best': True,
    'workers': 2,
    'every': 20,
    'split': True,
    'shifts': 1,
    'overlap': 0.25,
    'sdr': True,
    'metric': 'loss',
    'nonhq': None
}

svd = {
    'penalty': 0,
    'min_size': 0.1,
    'dim': 1,
    'niters': 2,
    'powm': False,
    'proba': 1,
    'conv_only': False,
    'convtr': False,
    'bs': 1
}

args = {
    "epochs": 100,
    "batch_size": 30,
    "max_batches": 2,
    'optim' : optim_params,
    'htdemucs' : ht_demucs_params,
    'quant' : {
        'diffq': None,
        'qat': None,
        'min_size': 0.2,
        'group_size': 8
    },
    'ema': {
        'epoch': [],
        'batch': []
    },
    'misc': misc,
    'test': test,
    'seed': 42,
    'debug': False,
    'valid_apply': True,
    'flag': 'None',
    'save_every': None,
    'weights': [1.],  # weights over each source for the training/valid loss.
    'svd': svd
}
args = DefaultMunch.fromDict(args)

def get_optimizer(model, args=args):
    seen_params = set()
    other_params = []
    groups = []
    for n, module in model.named_modules():
        if hasattr(module, "make_optim_group"):
            group = module.make_optim_group()
            params = set(group["params"])
            assert params.isdisjoint(seen_params)
            seen_params |= set(params)
            groups.append(group)
    for param in model.parameters():
        if param not in seen_params:
            other_params.append(param)
    groups.insert(0, {"params": other_params})
    parameters = groups
    if args.optim.optim == "adam":
        return torch.optim.Adam(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    elif args.optim.optim == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=args.optim.lr,
            betas=(args.optim.momentum, args.optim.beta2),
            weight_decay=args.optim.weight_decay,
        )
    else:
        raise ValueError("Invalid optimizer %s", args.optim.optimizer)


def get_model():
    extra = {
        'sources': ["piano"],
        'audio_channels': 1,
        'samplerate': 44100,
        'segment': 12,
    }

    return HTDemucs(**extra, **ht_demucs_params)

def get_datasets():
    train_dataset = PianoDataset(train=True, seq_len=44100*10, mono=True)
    val_dataset = PianoDataset( train=False, seq_len=44100*10, mono=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=4, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, num_workers=4, pin_memory=True)
    return {"train": train_dataloader, "valid": val_dataloader}


if __name__ == "__main__":
    model = get_model()
    optimizer = get_optimizer(model)
    dataloaders = get_datasets()
    solver = Solver(dataloaders, model, optimizer, args)
    solver.train()

