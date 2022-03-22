import typer
from pathlib import Path
from PIL import Image
from typing import List
from pathlib import Path

import torch
from torch import Tensor
import numpy as np
from torchvision import models, transforms
from torchvision.io import read_image

from .datasets import ImageDataset
from torch.utils.data import DataLoader

app = typer.Typer()

@app.command()
def cluster(
    inputs:List[Path],
): 
    # find images
    images = []
    for i in inputs:
        if i.is_dir():
            # TODO search directory
            continue
        else:
            images.append(i)


    dataset = ImageDataset(images)
    print(dataset[0].shape)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = models.vgg19(pretrained=True)

    with torch.no_grad():
        results = [model(batch) for batch in dataloader]
    results = torch.cat(results, dim=0)

    return 


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for input_file in images:
        # im = read_image(str(input_file))
        im = Image.open(input_file)
        # print(im)
        # im = Tensor(np.array(im))
        tensor = transform(im)
        print(tensor)

    # get model
    # model = models.resnet18(pretrained=True)
    model = models.vgg19(pretrained=True)
    batch = torch.unsqueeze(tensor, 0)
    result = model(batch)
    print(result.shape)
