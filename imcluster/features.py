import types
import enum
from typing import get_type_hints, List
import numpy as np
from PIL import Image
import torch
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.functional import normalize
from torch import nn
from rich.progress import track
from rich.console import Console
from img2vec_pytorch import Img2Vec

console = Console()

from .io import ImclusterIO


def torchvision_model_choices() -> List[str]:
    """
    Returns a list of function names in torchvision.models which can produce torch modules.
    """
    model_choices = []
    for item in dir(models):
        obj = getattr(models, item)

        # Only accept functions
        if isinstance(obj, types.FunctionType):

            # Only accept if the return value is a pytorch module
            hints = get_type_hints(obj)
            return_value = hints.get("return", "")
            if nn.Module in return_value.mro():
                model_choices.append(item)
    return model_choices


TorchvisionModelName = enum.Enum(
    "TorchvisionModelName",
    {model_name: model_name for model_name in torchvision_model_choices()},
)


def build_features(
    imcluster_io: ImclusterIO,
    model_name: TorchvisionModelName = "vgg19",
    force: bool = False,
):
    """
    Builds a list of feature vectors for all the images from a pretrained pytorch model.

    Saves results into a column with the same name as the torchvision model.
    """
    # Convert the enum value to its value if necessary
    if isinstance(model_name, TorchvisionModelName):
        model_name = model_name.value
    model_name = str(model_name)

    img_size = 224 # The minimum size for torchvision (https://pytorch.org/vision/stable/models.html)

    if not imcluster_io.has_column(model_name) or force:
        # if False:

        class ImageDataset(Dataset):
            def __init__(self, images):
                self.images = images
                # See note on normalization at https://pytorch.org/vision/stable/models.html
                self.transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                input_file = self.images[idx]
                im = Image.open(input_file)

                # enforce landscape rotation
                if im.width < im.height:
                    im = im.rotate(90)

                im = im.resize( (img_size,img_size) )

                image = self.transform(im)
                return image

        console.print("Setting up dataset")
        img2vec = Img2Vec(cuda=False)

        results = []
        for path in track(imcluster_io.images, description="Generating feature vectors:"):
            im = Image.open(path)
            
            # enforce landscape rotation
            if im.width < im.height:
                im = im.rotate(90)

            result = torch.flatten(img2vec.get_vec(im, tensor=True), start_dim=1)
            results.append(result)
        feature_vectors = torch.cat(results, dim=0)
        # print(feature_vectors.size())
        feature_vectors = normalize(feature_vectors, dim=0)
        feature_vectors = feature_vectors.cpu().detach().numpy()

        imcluster_io.save_column(
            model_name, [feature_vectors[x] for x in range(feature_vectors.shape[0])]
        )
    else:
        console.print("Using precomputed feature vectors")

        feature_vectors = np.array(imcluster_io.get_column(model_name).to_list())

    return feature_vectors
