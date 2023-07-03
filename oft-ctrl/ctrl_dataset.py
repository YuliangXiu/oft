import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepFashionDenseposeDataset(Dataset):
    def __init__(self, split='train', resolution=512):

        self.data = []
        self.split = split
        self.resolution = resolution

        with open(
            '../data/DeepFashion/{}/prompt_{}_blip.json'.format(split, split), 'rt'
        ) as f:    # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        prompt = item['prompt']

        source = Image.open(
            '../data/DeepFashion/{}/densepose/'.format(self.split) + source_filename[7:]
        ).convert("RGB")
        target = Image.open(
            '../data/DeepFashion/{}/color/'.format(self.split) + source_filename[7:]
        ).convert("RGB")

        image = self.image_transforms(target)
        conditioning_image = self.conditioning_image_transforms(source)

        return dict(
            pixel_value=image,
            conditioning_pixel_value=conditioning_image,
            caption=prompt,
        )


def collate_fn(examples, tokenizer):
    pixel_values = [example["pixel_value"] for example in examples]
    conditioning_pixel_values = [example["conditioning_pixel_value"] for example in examples]
    captions = [example["caption"] for example in examples]

    input_ids = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack(conditioning_pixel_values)
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format
                                                            ).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
    }

    return batch