import os

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


class InatDataset(Dataset):
    """
    Dataset for the iNaturalist challenge 2021.
    More info on:
      * https://github.com/visipedia/inat_comp/tree/master/2021
      * https://www.kaggle.com/c/inaturalist-2021/overview/description
    """

    def __init__(self, cfg, transform):
        self.image_path = cfg["images"]
        self.transform = transform

        self.coco = COCO(cfg["anno_json"])
        # Create the dataset based on image IDs (images only loaded on demand)
        self.ids = list(sorted(self.coco.getImgIds()))
        print("Loaded dataset with size {}".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image ID from the dataset at the given index
        image_id = self.ids[idx]
        # Load the annotation IDs for the given image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        # Load the actual annotations for the given IDs
        target = self.coco.loadAnns(annotation_ids)[0]
        # Query the label
        label = target["category_id"]

        # Query the image filename and other parameters
        image_filename = self.coco.loadImgs(image_id)[0]["file_name"]
        image_lat = self.coco.loadImgs(image_id)[0]["latitude"]
        image_lon = self.coco.loadImgs(image_id)[0]["longitude"]
        image_loc_uncert = self.coco.loadImgs(image_id)[0]["location_uncertainty"]

        # Load the image using PIL using the RGB color space
        image_path = os.path.join(self.image_path, image_filename)
        image = Image.open(image_path).convert("RGB")
        # Transform the image (augmentation, to Torch, ...)
        image = self.transform(image)

        return {
            "image": image,
            "label": label,
            # "metadata": {
            #     "lat": image_lat,
            #     "lon": image_lon,
            #     "loc_uncert": image_loc_uncert
            # }
        }
