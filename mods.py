from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from skimage.segmentation import slic
from skimage.util import img_as_float


class PILFilterTransform(object):
    """Apply a squared median filter on a PIL Image."""

    def __init__(self, filter_op='median', kernel_size=5):
        super(PILFilterTransform, self).__init__()
        self.filter = filter_op
        self.kernel_size = kernel_size

        if filter_op == 'median':
            self.pil_filter = ImageFilter.MedianFilter(self.kernel_size)
        elif filter_op == 'mean':
            self.pil_filter = ImageFilter.BoxBlur(self.kernel_size // 2)
    
    def __call__(self, pil_image):
        return pil_image.filter(self.pil_filter)

    def __repr__(self):
        return self.__class__.__name__ + f'(filter={self.filter}, kernel_size={self.kernel_size})'


class PILHistEqTransform(object):
    """ Apply Histogram Equalization on a PIL Image. """
    def __call__(self, pil_image):
        return ImageOps.equalize(pil_image)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PILJpegQualityTransform(object):
    """ Apply Jpeg Compression and Decompression on a PIL Image. """
    def __init__(self, quality=60):
        super(PILJpegQualityTransform, self).__init__()
        self.quality = quality

    def __call__(self, pil_image):
        encoded_jpeg = BytesIO()
        pil_image.save(encoded_jpeg, format='jpeg', quality=self.quality, subsampling=0)
        return Image.open(encoded_jpeg)

    def __repr__(self):
        return self.__class__.__name__ + f'(quality={self.quality})'


class PILCopyMoveTransform(object):
    """ Apply Copy-Move modification on a PIL Image. """
    def __init__(self, seed=42):
        super(PILCopyMoveTransform, self).__init__()
        self.seed = seed

    def __call__(self, pil_image):
        np_image = np.array(pil_image)

        # find superpixels
        segments = slic(img_as_float(np_image), n_segments=25, start_label=0)
        h, w = segments.shape

        # random put predictable per-image seed
        seed = np_image.sum().astype(int) + self.seed
        rng = np.random.default_rng(seed)

        # choose a random superpixel
        copy_segment_idx = rng.choice(np.unique(segments))
        yy, xx = np.where(segments == copy_segment_idx)

        ymin, ymax = yy.min(), yy.max() + 1
        xmin, xmax = xx.min(), xx.max() + 1

        sh = ymax - ymin
        sw = xmax - xmin

        # pick destination at random
        nyy = yy - ymin + rng.integers(h - sh, endpoint=True)
        nxx = xx - xmin + rng.integers(w - sw, endpoint=True)

        # copy-move
        np_image[nyy, nxx] = np_image[yy, xx]

        return Image.fromarray(np_image)

    def __repr__(self):
        return self.__class__.__name__ + f'(seed={self.seed})'


def get_modification_transform(kwargs):
    modification = kwargs['modification']

    if modification == 'filter':
        filter_op = kwargs['operation']
        window_size = kwargs['window-size']
        return PILFilterTransform(filter_op=filter_op, kernel_size=window_size)

    if modification == 'hist-eq':
        return PILHistEqTransform()

    if modification == 'jpeg':
        quality = kwargs['quality']
        return PILJpegQualityTransform(quality=quality)
    
    if modification == 'copy-move':
        seed = kwargs['seed']
        return PILCopyMoveTransform(seed=seed)
    
    raise NotImplementedError(f'Modification "{modification}" not implemented.')

        
def add_modification_argparse(parser):
    subparsers = parser.add_subparsers(dest='modification', help='type of image modifications to detect')

    filter_parser = subparsers.add_parser('filter')
    filter_parser.add_argument('operation', choices=('median', 'mean'), help='filter operation')
    filter_parser.add_argument('window-size', type=int, help='filter window size')

    hist_eq_parser = subparsers.add_parser('hist-eq')

    jpeg_parser = subparsers.add_parser('jpeg')
    jpeg_parser.add_argument('quality', type=int, help='jpeg compression quality [0, 100]')

    copymove_parser = subparsers.add_parser('copy-move')
    copymove_parser.add_argument('seed', type=int, help='random seed')


def get_modification_string(args):
    if args['modification'] == 'filter':
        op, w = args["operation"], args["window-size"]
        return f'{op}-{w}x{w}'

    if args['modification'] == 'hist-eq':
        return f'hist-eq'

    if args['modification'] == 'jpeg':
        quality = args['quality']
        return f'jpeg-{quality}'
    
    if args['modification'] == 'copy-move':
        seed = args['seed']
        return f'copymove-{seed}'
    
    return None