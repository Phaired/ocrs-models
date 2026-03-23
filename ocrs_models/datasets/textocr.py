"""
TextOCR dataset loader for text detection and recognition.

TextOCR contains ~28K images with ~900K word-level polygon annotations
from natural scenes. Annotations follow a COCO-like JSON format.

Dataset: https://textvqa.org/textocr/
License: CC BY 4.0
"""

import json
import os
from typing import Optional, cast

import torch
from torchvision.io import read_image, ImageReadMode, write_png
from torchvision.transforms.functional import resize

from .hiertext import DEFAULT_ALPHABET
from .util import SizedDataset, encode_text, generate_mask, transform_image


class TextOCR(SizedDataset):
    """
    TextOCR dataset for text detection training.

    Expects the dataset directory to contain:
    - train_val_images/: image directory (from train_val_images.zip)
    - TextOCR_0.1_train.json: training annotations
    - TextOCR_0.1_val.json: validation annotations
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        train=True,
        max_images: Optional[int] = None,
    ):
        super().__init__()

        self._root_dir = root_dir
        self.transform = transform

        if train:
            ann_file = os.path.join(root_dir, "TextOCR_0.1_train.json")
        else:
            ann_file = os.path.join(root_dir, "TextOCR_0.1_val.json")

        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"TextOCR annotations not found at {ann_file}. "
                "Download from https://textvqa.org/textocr/"
            )

        with open(ann_file) as f:
            data = json.load(f)

        self._imgs = data["imgs"]
        self._anns = data["anns"]
        self._img_to_anns = data.get("imgToAnns", {})

        # Build image list
        self._image_ids = list(self._imgs.keys())

        if max_images:
            self._image_ids = self._image_ids[:max_images]

    def __len__(self):
        return len(self._image_ids)

    def _get_image_path(self, img_info: dict) -> str:
        """Resolve image path from image info."""
        filename = img_info.get("file_name", img_info.get("id", ""))
        # Strip set prefix (e.g. "train/abc.jpg" -> "abc.jpg")
        basename = os.path.basename(filename)
        for subdir in ["train_images", "train_val_images", ""]:
            for name in [filename, basename]:
                path = os.path.join(self._root_dir, subdir, name)
                if os.path.exists(path):
                    return path
        return os.path.join(self._root_dir, "train_images", basename)

    def __getitem__(self, idx: int):
        img_id = self._image_ids[idx]
        img_info = self._imgs[img_id]
        img_path = self._get_image_path(img_info)

        img = read_image(img_path, ImageReadMode.GRAY)
        img = transform_image(img)
        _, img_height, img_width = img.shape

        # Get annotations for this image
        ann_ids = self._img_to_anns.get(img_id, [])
        polys = []
        for ann_id in ann_ids:
            if ann_id not in self._anns:
                continue
            ann = self._anns[ann_id]

            # Skip illegible text
            text = ann.get("utf8_string", "")
            if text == ".":
                continue

            # Extract polygon points
            points = ann.get("points", [])
            if len(points) >= 8:
                # Points are [x1,y1,x2,y2,...] flat list
                poly = [
                    (int(points[i]), int(points[i + 1]))
                    for i in range(0, len(points), 2)
                ]
                polys.append(poly)
            elif "bbox" in ann:
                # Fall back to bbox [x1, y1, x2, y2]
                x1, y1, x2, y2 = ann["bbox"]
                polys.append([
                    (int(x1), int(y1)),
                    (int(x2), int(y1)),
                    (int(x2), int(y2)),
                    (int(x1), int(y2)),
                ])

        text_mask = generate_mask(img_width, img_height, polys)

        if self.transform:
            combined = torch.cat([img, text_mask.unsqueeze(0)], dim=0)
            combined = self.transform(combined)
            img = combined[0:1]
            text_mask = combined[1:2]
        else:
            text_mask = text_mask.unsqueeze(0)

        return {
            "path": img_path,
            "image": img,
            "text_mask": text_mask,
        }


class TextOCRRecognition(SizedDataset):
    """
    TextOCR dataset for text recognition training.

    Yields cropped word images with their text labels. Words are cropped
    from full images using bounding boxes derived from polygon annotations.
    """

    def __init__(
        self,
        root_dir: str,
        train=True,
        transform=None,
        max_images=None,
        alphabet: Optional[list[str]] = None,
        output_height: int = 64,
    ):
        super().__init__()

        if alphabet is None:
            alphabet = [c for c in DEFAULT_ALPHABET]
        self.alphabet = cast(list[str], alphabet)

        self._root_dir = root_dir
        self.transform = transform
        self.output_height = output_height

        if train:
            ann_file = os.path.join(root_dir, "TextOCR_0.1_train.json")
        else:
            ann_file = os.path.join(root_dir, "TextOCR_0.1_val.json")

        with open(ann_file) as f:
            data = json.load(f)

        self._imgs = data["imgs"]

        # Build flat list of (img_id, ann) for words with valid text
        self._words = []
        for ann_id, ann in data["anns"].items():
            text = ann.get("utf8_string", "")
            if not text or text == ".":
                continue
            img_id = ann.get("image_id", "")
            if img_id not in self._imgs:
                continue
            points = ann.get("points", [])
            if len(points) < 8 and "bbox" not in ann:
                continue
            self._words.append((img_id, ann))

        if max_images:
            self._words = self._words[:max_images]

        self._cache_dir = os.path.join(root_dir, "word-cache")

    def __len__(self):
        return len(self._words)

    def _get_image_path(self, img_info: dict) -> str:
        filename = img_info.get("file_name", img_info.get("id", ""))
        basename = os.path.basename(filename)
        for subdir in ["train_images", "train_val_images", ""]:
            for name in [filename, basename]:
                path = os.path.join(self._root_dir, subdir, name)
                if os.path.exists(path):
                    return path
        return os.path.join(self._root_dir, "train_images", basename)

    def _get_bbox(self, ann: dict) -> tuple[int, int, int, int]:
        points = ann.get("points", [])
        if len(points) >= 8:
            xs = [int(points[i]) for i in range(0, len(points), 2)]
            ys = [int(points[i]) for i in range(1, len(points), 2)]
            return min(xs), min(ys), max(xs), max(ys)
        elif "bbox" in ann:
            x1, y1, x2, y2 = ann["bbox"]
            return int(x1), int(y1), int(x2), int(y2)
        return 0, 0, 0, 0

    def __getitem__(self, idx: int):
        img_id, ann = self._words[idx]
        img_info = self._imgs[img_id]
        text = ann["utf8_string"]

        min_x, min_y, max_x, max_y = self._get_bbox(ann)

        # Load from cache or crop
        ann_id = ann.get("id", str(idx))
        cache_path = os.path.join(self._cache_dir, f"{ann_id}.png")

        if not os.path.exists(cache_path):
            img_path = self._get_image_path(img_info)
            img = read_image(img_path, ImageReadMode.GRAY)
            _, img_h, img_w = img.shape

            min_x = max(0, min(min_x, img_w - 1))
            max_x = max(min_x + 1, min(max_x, img_w))
            min_y = max(0, min(min_y, img_h - 1))
            max_y = max(min_y + 1, min(max_y, img_h))

            word_img = img[:, min_y:max_y, min_x:max_x]

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = cache_path + ".tmp"
            write_png(word_img, tmp_path)
            os.rename(tmp_path, cache_path)

        line_img = transform_image(read_image(cache_path, ImageReadMode.GRAY))
        _, line_height, line_width = line_img.shape

        # Skip tiny crops that cause padding errors
        if line_height < 5 or line_width < 5:
            line_img = torch.full((1, self.output_height, 10), -0.5)
            text = "?"

        if self.transform and line_height >= 5 and line_width >= 5:
            line_img = self.transform(line_img)
            line_img = line_img.clamp(-0.5, 0.5)
            _, line_height, line_width = line_img.shape

        aspect_ratio = line_width / max(line_height, 1)
        output_width = min(800, max(10, int(self.output_height * aspect_ratio)))
        line_img = resize(line_img, [self.output_height, output_width], antialias=True)

        text_seq = encode_text(text, self.alphabet, unknown_char="?")

        return {
            "image_id": img_id,
            "image": line_img,
            "text_seq": text_seq,
            "text": text,
        }
