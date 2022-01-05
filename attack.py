import argparse
from pathlib import Path

import foolbox as fb
import pandas as pd
from mods import get_modification_transform
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import get_tinyimagenet
from train import LitODENet


class OneToTwoOutputs(torch.nn.Module):
    """ foolbox needs a two-output model for binary classification. """
    def __init__(self, model):
        super(OneToTwoOutputs, self).__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        y = torch.cat((-y, y), dim=1)
        return y


def main(args):
    ckpts = args.run_dir.glob('lightning_logs/version_*/checkpoints/*.ckpt')
    ckpt = sorted(ckpts, reverse=True, key=lambda p: p.stat().st_mtime)[0]
    print('Loading ckpt:', ckpt)

    adv_dir = args.run_dir / 'advs'
    adv_dir.mkdir(exist_ok=True)

    lit_model = LitODENet.load_from_checkpoint(ckpt)

    def set_ode_solver_tolerance(tol):
        lit_model.model.tol = tol

    model = OneToTwoOutputs(lit_model.model)
    model.eval()

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.L2CarliniWagnerAttack(binary_search_steps=7, steps=1000)

    modification = get_modification_transform(**vars(args))
    print(modification)
    
    dataset = get_tinyimagenet(modification, num_images=args.num_images, split='test')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    epsilons = [
        0.0,
        0.0001,
        0.0003,
        0.0005,
        0.001,
        0.003,
        0.005,
        0.01,
        0.03,
        0.05,
        0.1,
        0.3,
        0.5,
        1.0,
        3.0,
        5.0,
        10.0,
        30.0,
        50.0,
    ]

    # the tolerance used during train
    train_tol = lit_model.model.tol

    solver_tolerances = [
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1e0,
        1e1,
        1e2,
        1e3
    ]

    for i, (image, label) in enumerate(tqdm(dataloader)):
        adv_cache = adv_dir / f'{i}.pth'
        res_cache = adv_dir / f'{i}.csv'

        if res_cache.exists():
            continue
            
        if not adv_cache.exists():
            image = image.to(fmodel.device)
            label = label.to(fmodel.device)

            set_ode_solver_tolerance(train_tol)
            raw_advs, clipped_advs, success = attack(fmodel, image, label, epsilons=epsilons)

            advs = [
                {
                    'sample_idx': i,
                    'label': label.item(),
                    'attack_tol': train_tol,
                    'eps': eps,
                    'raw_adv': raw_adv,
                    'clipped_adv': clipped_adv,
                    'success': succ,
                } for eps, raw_adv, clipped_adv, succ
                    in zip(epsilons, raw_advs, clipped_advs, success)
            ]

            torch.save(advs, adv_cache)

        else:
            advs = torch.load(adv_cache)
            clipped_advs = [a['clipped_adv'] for a in advs]
            # success = [a['success'] for a in advs]

        adv_inputs = torch.cat(clipped_advs, dim=0)

        results = []
        for tol in tqdm(solver_tolerances, leave=False):

            set_ode_solver_tolerance(tol)
            adv_outputs = fmodel(adv_inputs)[:, 1]
            adv_scores = torch.sigmoid(adv_outputs)

            results.extend([
                {
                    'sample_idx': i,
                    'label': label.item(),
                    'tol': tol,
                    'eps': eps,
                    'score': score.item(),
                } for eps, score in zip(epsilons, adv_scores)
            ])

        results = pd.DataFrame(results)
        results.to_csv(res_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attack Forensic Neural ODE with L2 Carlini Wagner')
    parser.add_argument('run_dir', type=Path, help='path to root dir of trained model')
    parser.add_argument('-n', '--num-images', type=int, default=1_000, help='number of images to process')

    subparsers = parser.add_subparsers(dest='modification', help='type of image modifications to detect')

    filter_parser = subparsers.add_parser('filter')
    filter_parser.add_argument('operation', choices=('median', 'mean'), help='filter operation')
    filter_parser.add_argument('window-size', type=int, help='filter window size')

    args = parser.parse_args()
    main(args)