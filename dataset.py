from torch.utils.data import Dataset
import json
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, data_dir, part):
        super().__init__()
        self.data_dir = data_dir
        self.part = part

        self.image_paths = []
        self.points = []

        self.setup(part)

    def __getitem__(self, index):

        image = self.read_image(index)
        points = self.points[index]

        patch_image = image[points[0]-:points[0]+, points[1]-:points[1]+]

    def setup(self, part):
        json_path = os.path.join(self.data_dir, "annotation.json")
        with open(json_path, "r") as f:
            json_data = json.load(f)

        image_list = json_data["file_name"]
        for img in image_list:
            image_path = os.path.join(self.data_dir, img)
            image_part_point = json_data[img][part]

            self.image_paths.append(image_path)
            self.points.append(image_part_point)

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.oepn(image_path).convert("RGB")
