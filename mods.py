from PIL import ImageFilter

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
    

def get_modification_transform(**kwargs):
    modification = kwargs['modification']

    if modification == 'filter':
        filter_op = kwargs['operation']
        window_size = kwargs['window-size']
        return PILFilterTransform(filter_op=filter_op, kernel_size=window_size)
    
    raise NotImplementedError(f'Modification "{modification}" not implemented.')

        
