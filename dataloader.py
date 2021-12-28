from pathlib import Path

from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor, Compose


class TinyImageNet200(Dataset):
    def __init__(self, root='data/tiny-imagenet-200', split='train', num_images=10_000, target=0, transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target = target
        
        assert (num_images % 200 == 0) or (split == 'test'), "'num_images' must be a multiple of 200 when 'split' != 'test'"
        num_images_per_class = num_images / 200
        
        if split == 'train':
            labels = pd.read_csv(self.root / 'wnids.txt', header=None, squeeze=True).tolist()
            annot_files = [self.root / 'train' / label / f'{label}_boxes.txt' for label in labels]
            annots = [pd.read_csv(a, sep='\t', usecols=(0,), names=('filename',)) for a in annot_files]
            annot = pd.concat(annots, ignore_index=True)
            annot['label'] = annot['filename'].map(lambda x: x.split('_')[0])
            annot = annot.groupby('label').head(num_images_per_class)
            self.image_paths = ('train/' + annot['label'] + '/images/' + annot['filename']).tolist()

        elif split == 'val':
            annot_path = self.root / 'val' / 'val_annotations.txt'
            annot = pd.read_csv(annot_path, sep='\t', usecols=(0, 1), names=('filename', 'label'))
            annot = annot.groupby('label').head(num_images_per_class)
            self.image_paths = ('val/images/' + annot['filename']).tolist()
        
        elif split == 'test':
            self.image_paths = [f'test/images/test_{i}.JPEG' for i in range(num_images)]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.root / self.image_paths[index]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.target


class InterleaveDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
    
    def __len__(self):
        return sum(map(len, self.datasets))
    
    def __getitem__(self, index):
        dataset_idx = index % len(self.datasets)
        sample_idx = index // len(self.datasets)
        return self.datasets[dataset_idx][sample_idx]


def get_tinyimagenet(modif_transform, num_images=10_000, split='val'):
    modif_transform = Compose((modif_transform, ToTensor()))
    clean = TinyImageNet200(num_images=num_images, split=split, target=0, transform=ToTensor())
    modif = TinyImageNet200(num_images=num_images, split=split, target=1, transform=modif_transform)
    dataset = InterleaveDataset((clean, modif))
    return dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    from mods import PILFilterTransform
    
    clean_transform = ToTensor()
    clean_dataset = TinyImageNet200(num_images_per_class=10, split='val', target=0, transform=clean_transform)
    clean_dataloader = DataLoader(clean_dataset, batch_size=16, shuffle=False)
    x, y = next(iter(clean_dataloader))
    save_image(x, 'clean.png', nrow=4)

    for filter in ('mean', 'median'):
        for kernel_size in (3, 5, 7):
            modif_transform = Compose((PILFilterTransform(filter, kernel_size), ToTensor()))
            modif_dataset = TinyImageNet200(num_images_per_class=10, split='val', target=1, transform=modif_transform)
            modif_dataloader = DataLoader(modif_dataset, batch_size=16, shuffle=False)
            x, y = next(iter(modif_dataloader))
            save_image(x, f'{filter}{kernel_size}.png', nrow=4)

