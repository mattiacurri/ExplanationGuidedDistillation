"""Dataset helpers for Mini-ImageNet and related transforms."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict
from PIL import Image
from timm.data import ImageNetInfo
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils.model_utils import resolve_processor_output_size

MINI_IMAGENET_SIMPLE_LABELS: dict[int, str] = {
    0: "house finch",
    1: "American robin",
    2: "triceratops",
    3: "green mamba",
    4: "harvestman",
    5: "toucan",
    6: "goose",
    7: "jellyfish",
    8: "nematode",
    9: "red king crab",
    10: "dugong",
    11: "Treeing Walker Coonhound",
    12: "Ibizan Hound",
    13: "Saluki",
    14: "Golden Retriever",
    15: "Gordon Setter",
    16: "Komondor",
    17: "Boxer",
    18: "Tibetan Mastiff",
    19: "French Bulldog",
    20: "Alaskan Malamute",
    21: "Dalmatian",
    22: "Newfoundland",
    23: "Miniature Poodle",
    24: "Alaskan tundra wolf",
    25: "African wild dog",
    26: "Arctic fox",
    27: "lion",
    28: "meerkat",
    29: "ladybug",
    30: "rhinoceros beetle",
    31: "ant",
    32: "black-footed ferret",
    33: "three-toed sloth",
    34: "rock beauty",
    35: "aircraft carrier",
    36: "waste container",
    37: "barrel",
    38: "beer bottle",
    39: "bookstore",
    40: "cannon",
    41: "carousel",
    42: "carton",
    43: "catamaran",
    44: "chime",
    45: "clogs",
    46: "cocktail shaker",
    47: "combination lock",
    48: "crate",
    49: "cuirass",
    50: "dishcloth",
    51: "dome",
    52: "electric guitar",
    53: "filing cabinet",
    54: "fire screen sheet",
    55: "frying pan",
    56: "garbage truck",
    57: "barrette",
    58: "holster",
    59: "horizontal bar",
    60: "hourglass",
    61: "iPod",
    62: "lipstick",
    63: "miniskirt",
    64: "missile",
    65: "mixing bowl",
    66: "oboe",
    67: "organ",
    68: "parallel bars",
    69: "pencil case",
    70: "photocopier",
    71: "poncho",
    72: "prayer rug",
    73: "reel",
    74: "school bus",
    75: "scoreboard",
    76: "slot machine",
    77: "snorkel",
    78: "solar thermal collector",
    79: "spider web",
    80: "stage",
    81: "tank",
    82: "front curtain",
    83: "tile roof",
    84: "tobacco shop",
    85: "unicycle",
    86: "upright piano",
    87: "vase",
    88: "wok",
    89: "split-rail fence",
    90: "yawl",
    91: "traffic sign",
    92: "consomme",
    93: "trifle",
    94: "hot dog",
    95: "orange",
    96: "cliff",
    97: "coral reef",
    98: "bolete",
    99: "ear of corn",
}


def build_processor_transform(
    processor: Any,
    *,
    augment: bool,
) -> transforms.Compose:
    """Build torchvision transforms aligned with the ViT processor stats."""
    size = resolve_processor_output_size(processor)
    normalize = transforms.Normalize(
        mean=processor.image_mean,
        std=processor.image_std,
    )

    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class MiniImageNetDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch dataset wrapper around a HuggingFace image dataset."""

    def __init__(self, hf_dataset, transform: transforms.Compose) -> None:
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = int(item["label"])
        pixel_values = self.transform(image)
        return pixel_values, label


class MiniImageNetRawDataset(Dataset[tuple[Image.Image, int]]):
    """PyTorch dataset wrapper returning raw RGB images and labels."""

    def __init__(self, hf_dataset) -> None:
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = int(item["label"])
        return image, label


class PreprocessedMiniImageNetDataset(Dataset[dict[str, Any]]):
    """Local fixed-size Mini-ImageNet split saved as torch records."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        split: str,
        files: list[Path] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.files = files or sorted(self.split_dir.glob("sample_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"Nessun sample preprocessato trovato in {self.split_dir}"
            )
        self.metadata = self._load_metadata()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int | str) -> dict[str, Any] | list[int]:
        if isinstance(idx, str):
            if idx != "label":
                raise KeyError(idx)
            return [
                int(torch.load(path, map_location="cpu")["label"])
                for path in self.files
            ]

        sample = torch.load(self.files[idx], map_location="cpu")
        image_tensor = sample["image"]
        if image_tensor.dtype != torch.uint8:
            image_tensor = image_tensor.to(torch.uint8)
        image_array = image_tensor.permute(1, 2, 0).numpy()
        return {
            "image": Image.fromarray(image_array, mode="RGB"),
            "label": int(sample["label"]),
            "source_index": int(sample.get("source_index", idx)),
            "original_size": tuple(int(v) for v in sample.get("original_size", ())),
        }

    def select(self, indices: Sequence[int]) -> "PreprocessedMiniImageNetDataset":
        selected_files = [self.files[int(index)] for index in indices]
        return PreprocessedMiniImageNetDataset(
            self.data_dir,
            split=self.split,
            files=selected_files,
        )

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))


def preprocess_mini_imagenet_split(
    dataset: Any,
    *,
    output_dir: str | Path,
    split: str,
    image_size: int = 392,
    overwrite: bool = False,
    label_names: Sequence[str] | None = None,
) -> int:
    """Save one split as fixed-size RGB uint8 tensors."""
    if image_size <= 0:
        raise ValueError("image_size deve essere maggiore di 0")
    root = Path(output_dir)
    split_dir = root / split
    existing_files = sorted(split_dir.glob("sample_*.pt")) if split_dir.exists() else []
    if existing_files and not overwrite:
        raise FileExistsError(
            f"{split_dir} contiene gia' {len(existing_files)} sample. "
            "Usa --overwrite oppure una nuova directory."
        )

    split_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for sample_file in existing_files:
            sample_file.unlink()

    resample = getattr(Image, "Resampling", Image).BICUBIC
    for idx in range(len(dataset)):
        item = dataset[idx]
        image = item["image"].convert("RGB")
        original_size = tuple(int(v) for v in image.size)
        resized = image.resize((image_size, image_size), resample=resample)
        array = np.asarray(resized, dtype=np.uint8)
        image_tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
        torch.save(
            {
                "image": image_tensor,
                "label": int(item["label"]),
                "source_index": int(idx),
                "original_size": original_size,
            },
            split_dir / f"sample_{idx:08d}.pt",
        )

    _write_preprocessed_metadata(
        root,
        split=split,
        image_size=image_size,
        num_samples=len(dataset),
        label_names=label_names,
    )
    return len(dataset)


def _write_preprocessed_metadata(
    root: Path,
    *,
    split: str,
    image_size: int,
    num_samples: int,
    label_names: Sequence[str] | None,
) -> None:
    metadata_path = root / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {"splits": {}}
    )
    metadata["image_size"] = int(image_size)
    if label_names is not None:
        metadata["label_names"] = [str(label) for label in label_names]
    metadata.setdefault("splits", {})[split] = int(num_samples)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_preprocessed_mini_imagenet(
    data_dir: str | Path,
    *,
    splits: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, PreprocessedMiniImageNetDataset]:
    """Load available preprocessed Mini-ImageNet splits."""
    root = Path(data_dir)
    dataset: dict[str, PreprocessedMiniImageNetDataset] = {}
    for split in splits:
        split_dir = root / split
        if split_dir.exists():
            dataset[split] = PreprocessedMiniImageNetDataset(root, split=split)
    if not dataset:
        raise FileNotFoundError(f"Nessuno split preprocessato trovato in {root}")
    return dataset


def build_label_maps(
    hf_dataset: DatasetDict | Mapping[str, Any],
) -> tuple[dict[int, str], dict[str, int]]:
    """Build id2label and label2id from a HuggingFace DatasetDict."""
    train_split = hf_dataset["train"]
    metadata = getattr(train_split, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("label_names"):
        names = [str(name) for name in metadata["label_names"]]
        id2label = {idx: name for idx, name in enumerate(names)}
        label2id = {name: idx for idx, name in id2label.items()}
        return id2label, label2id

    features = hf_dataset["train"].features["label"]
    id2label = {idx: name for idx, name in enumerate(features.names)}
    label2id = {name: idx for idx, name in enumerate(features.names)}
    return id2label, label2id


def build_mini_to_imagenet_mapping(label_names: Sequence[str]) -> dict[int, int]:
    """Map Mini-ImageNet label indices to ImageNet-1k indices via WNID."""
    info = ImageNetInfo()
    imagenet_wnids = info.label_names()
    wnid_to_imagenet_idx = {wnid: idx for idx, wnid in enumerate(imagenet_wnids)}

    mapping: dict[int, int] = {}
    for mini_idx, wnid in enumerate(label_names):
        if wnid not in wnid_to_imagenet_idx:
            msg = (
                "Mini-ImageNet class "
                f"{mini_idx} with wnid '{wnid}' is not in ImageNet-1k"
            )
            raise ValueError(msg)
        mapping[mini_idx] = wnid_to_imagenet_idx[wnid]

    return mapping


_MINI_IMAGENET_WNIDS: tuple[str, ...] = (
    "n01532829",
    "n01558993",
    "n01704323",
    "n01749939",
    "n01770081",
    "n01843383",
    "n01855672",
    "n01910747",
    "n01930112",
    "n01981276",
    "n02074367",
    "n02089867",
    "n02091244",
    "n02091831",
    "n02099601",
    "n02101006",
    "n02105505",
    "n02108089",
    "n02108551",
    "n02108915",
    "n02110063",
    "n02110341",
    "n02111277",
    "n02113712",
    "n02114548",
    "n02116738",
    "n02120079",
    "n02129165",
    "n02138441",
    "n02165456",
    "n02174001",
    "n02219486",
    "n02443484",
    "n02457408",
    "n02606052",
    "n02687172",
    "n02747177",
    "n02795169",
    "n02823428",
    "n02871525",
    "n02950826",
    "n02966193",
    "n02971356",
    "n02981792",
    "n03017168",
    "n03047690",
    "n03062245",
    "n03075370",
    "n03127925",
    "n03146219",
    "n03207743",
    "n03220513",
    "n03272010",
    "n03337140",
    "n03347037",
    "n03400231",
    "n03417042",
    "n03476684",
    "n03527444",
    "n03535780",
    "n03544143",
    "n03584254",
    "n03676483",
    "n03770439",
    "n03773504",
    "n03775546",
    "n03838899",
    "n03854065",
    "n03888605",
    "n03908618",
    "n03924679",
    "n03980874",
    "n03998194",
    "n04067472",
    "n04146614",
    "n04149813",
    "n04243546",
    "n04251144",
    "n04258138",
    "n04275548",
    "n04296562",
    "n04389033",
    "n04418357",
    "n04435653",
    "n04443257",
    "n04509417",
    "n04515003",
    "n04522168",
    "n04596742",
    "n04604644",
    "n04612504",
    "n06794110",
    "n07584110",
    "n07613480",
    "n07697537",
    "n07747607",
    "n09246464",
    "n09256479",
    "n13054560",
    "n13133613",
)


def get_mini_simple_label(imagenet_class_id: int) -> str | None:
    """Return the simplified Mini-ImageNet label for an ImageNet-1k class id.

    Returns None if the class is not part of Mini-ImageNet.
    """
    mini_to_imagenet = build_mini_to_imagenet_mapping(_MINI_IMAGENET_WNIDS)
    imagenet_to_mini = {v: k for k, v in mini_to_imagenet.items()}

    mini_idx = imagenet_to_mini.get(imagenet_class_id)
    if mini_idx is None:
        return None
    return MINI_IMAGENET_SIMPLE_LABELS.get(mini_idx)
