import numpy as np
from rich.console import Console
console = Console()
from .io import ImclusterIO


def build_features(imcluster_io:ImclusterIO, model_name:str="vgg19", force:bool=False):
    """
    Builds a list of feature vectors for all the images from a pretrained pytorch model.

    Saves results into a column named 'features'.
    """
    column_name = 'features'
    if not imcluster_io.has_column(column_name) or force:
        import torch
        from torchvision import models
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset
        from torchvision import transforms
        from PIL import Image

        class ImageDataset(Dataset):
            def __init__(self, images):
                self.images = images
                # See note on normalization at https://pytorch.org/vision/stable/models.html
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                input_file = self.images[idx]
                im = Image.open(input_file)
                
                # enforce landscape rotation
                if im.width < im.height:
                    im = im.rotate(90)

                image = self.transform(im)
                return image   

        console.print("Generating feature vectors")
        dataset = ImageDataset(imcluster_io.images)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model_class = getattr(models, model_name)
        model = model_class(pretrained=True)

        with torch.no_grad():
            results = [model(batch) for batch in dataloader]
        feature_vectors = torch.cat(results, dim=0).cpu().detach().numpy()

        imcluster_io.save_column(column_name, [feature_vectors[x] for x in range(feature_vectors.shape[0])])
    else:
        console.print("Using precomputed feature vectors")
        feature_vectors = np.array(imcluster_io.get_column(column_name).to_list())

    return feature_vectors