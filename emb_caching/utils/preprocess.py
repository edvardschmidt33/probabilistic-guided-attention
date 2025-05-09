from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def preprocess():
    return Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        lambda image: image.convert('RGB'),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)),
    ])
