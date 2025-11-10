from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == "__main__":
    data_folder = "../../../fsaf/h2-data/n02124075/"
    img_paths = []
    import os
    for file in os.listdir(data_folder):
        if file.endswith('.JPEG'):
            img_paths.append(os.path.join(data_folder, file))
    from torchvision.transforms import v2
    transform = v2.Compose([
        v2.ToTensor(), # (H, W, C) -> (C, H, W), values in [0,1]
        v2.Resize((64, 64)),  # resize to 64x64
    ])
    ds = ImageDataset(img_paths, transform=transform)
    import matplotlib.pyplot as plt
    for _ in range(5):
        plt.imshow(ds[_].permute(1, 2, 0))
        plt.show()


    