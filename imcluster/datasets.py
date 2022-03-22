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
        image = self.transform(im)
        return image

