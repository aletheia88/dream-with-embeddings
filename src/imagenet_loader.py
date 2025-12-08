from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from typing import Dict, Iterator, Optional, Tuple
import io
import tarfile
import torch


IMG_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")


def is_image_member(m: tarfile.TarInfo) -> bool:
    return m.isfile() and m.name.lower().endswith(IMG_EXTS)


def load_devkit_wnid_to_idx(devkit_dir: Path) -> Dict[str, int]:
    """
    Build the canonical mapping WNID -> 0..999 from the ILSVRC2012 devkit.
    Preferred source: data/meta.mat (or meta_clsloc.mat).
    Fallback: data/map_clsloc.txt (if it exists).

    Returns:
        dict: { 'n01440764': 0, ..., 'n15075141': 999 }
    """
    data_dir = Path(devkit_dir) / "data"

    # --- Preferred: meta.mat / meta_clsloc.mat
    meta_path = None
    for cand in ("meta.mat", "meta_clsloc.mat"):
        p = data_dir / cand
        if p.exists():
            meta_path = p
            break

    if meta_path is not None:
        # meta['synsets'] is a struct array with fields including 'WNID' and 'ILSVRC2012_ID'
        meta = loadmat(str(meta_path), squeeze_me=True, struct_as_record=False)
        synsets = meta["synsets"]
        wnid_to_idx = {}

        # synsets is typically a numpy object array of length >= 1000.
        # We keep only those with ILSVRC2012_ID in 1..1000 (the CLS-LOC subset),
        # then convert to 0-based.
        def _get(field):
            # helper to make attribute/field access robust
            return getattr(s, field) if hasattr(s, field) else s.__dict__[field]

        for s in synsets:
            # Some entries are non-CLS-LOC; we only keep the 1000-class subset
            ilsvrc_id = (
                int(_get("ILSVRC2012_ID")) if hasattr(s, "ILSVRC2012_ID") else None
            )
            if ilsvrc_id is None:
                continue
            wnid = str(_get("WNID"))
            idx0 = ilsvrc_id - 1
            wnid_to_idx[wnid] = idx0

        if len(wnid_to_idx) != 1000:
            raise ValueError(
                f"Expected 1000 classes from {meta_path}, got {len(wnid_to_idx)}."
            )
        return wnid_to_idx


def build_fallback_wnid_to_idx(train_dir: Path) -> Dict[str, int]:
    """
    Stable fallback mapping when no devkit is supplied:
    sort shard WNIDs lexicographically and assign 0..999.
    Note: This will NOT match the canonical ImageNet indices.
    """
    wnids = sorted(p.stem for p in train_dir.glob("*.tar"))
    if len(wnids) != 1000:
        raise ValueError(
            f"Expected 1000 train shards, found {len(wnids)} in {train_dir}"
        )
    return {w: i for i, w in enumerate(wnids)}


class ImageNetTrainTarDataset(IterableDataset):
    """
    Streams ImageNet train images directly from per-class tar shards.
    Label is inferred from the shard WNID -> class index.
    """

    def __init__(
        self,
        train_dir: Path,
        transform: transforms.Compose,
        wnid_to_idx: Optional[Dict[str, int]] = None,
        shuffle_images_within_shard: bool = False,
        shuffle_shards: bool = False,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.train_dir = Path(train_dir)
        self.shard_paths = sorted(self.train_dir.glob("*.tar"))
        if len(self.shard_paths) != 1000:
            raise ValueError(
                f"Expected 1000 tar shards under {train_dir}, found {len(self.shard_paths)}"
            )

        self.transform = transform
        self.base_seed = int(base_seed)
        self.shuffle_images_within_shard = bool(shuffle_images_within_shard)
        self.shuffle_shards = bool(shuffle_shards)
        self._epoch = 0

        # Build label mapping
        if wnid_to_idx is None:
            self.wnid_to_idx = build_fallback_wnid_to_idx(self.train_dir)
        else:
            self.wnid_to_idx = dict(wnid_to_idx)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _iter_one_shard(
        self, tar_path: Path, label: int, rng: torch.Generator
    ) -> Iterator[Tuple[torch.Tensor, int]]:
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if is_image_member(m)]
            if self.shuffle_images_within_shard:
                perm = torch.randperm(len(members), generator=rng).tolist()
                members = [members[i] for i in perm]

            for m in members:
                ef = tar.extractfile(m)
                if ef is None:
                    continue
                try:
                    with Image.open(io.BytesIO(ef.read())) as pil:
                        img = pil.convert("RGB")
                except Exception:
                    # Corrupt image; skip
                    continue
                yield self.transform(img), label

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        # deterministic per-epoch seed (different across workers)
        rng = torch.Generator()
        rng.manual_seed(self.base_seed + 131071 * (self._epoch + 1) + worker_id)

        # Optional shard shuffling (but keep worker partition deterministic)
        shard_paths = list(self.shard_paths)
        if self.shuffle_shards:
            perm = torch.randperm(len(shard_paths), generator=rng).tolist()
            shard_paths = [shard_paths[i] for i in perm]

        # split shards among workers
        shard_paths = shard_paths[worker_id::num_workers]

        for tar_path in shard_paths:
            wnid = tar_path.stem  # e.g., 'n01440764'
            if wnid not in self.wnid_to_idx:
                # Skip unexpected shard
                continue
            label = self.wnid_to_idx[wnid]
            yield from self._iter_one_shard(tar_path, label, rng)


class ImageNetValTarDataset(IterableDataset):
    """
    Streams ImageNet validation images from a single tar and pairs with labels
    from ILSVRC2012_validation_ground_truth.txt (1..1000 -> converted to 0..999).
    """

    def __init__(
        self,
        val_tar: Path,
        val_gt_txt: Path,
        transform: transforms.Compose,
        shuffle: bool = False,
        base_seed: int = 0,
    ) -> None:
        super().__init__()
        self.val_tar = Path(val_tar)
        self.val_gt_txt = Path(val_gt_txt)
        self.transform = transform
        self.shuffle = bool(shuffle)
        self.base_seed = int(base_seed)
        self._epoch = 0

        # Read ground-truth (1..1000) -> make 0-based
        with open(self.val_gt_txt, "r") as f:
            self.labels_0based = [int(x.strip()) - 1 for x in f if x.strip()]
        if len(self.labels_0based) != 50000:
            raise ValueError(
                f"Expected 50k validation labels, found {len(self.labels_0based)}"
            )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        rng = torch.Generator()
        rng.manual_seed(self.base_seed + 524287 * (self._epoch + 1) + worker_id)

        with tarfile.open(self.val_tar, "r") as tar:
            members = [m for m in tar.getmembers() if is_image_member(m)]

            # Validation filenames are ILSVRC2012_val_00000001.JPEG ... 00050000.JPEG.
            # We need them in lexical order to align with the GT file.
            members.sort(key=lambda m: m.name)

            # Worker split over images (not shards)
            members = members[worker_id::num_workers]

            if self.shuffle:
                perm = torch.randperm(len(members), generator=rng).tolist()
                members = [members[i] for i in perm]

            # We need to index into the global 50k labels; compute positions:
            # The global position of our first member is its index in the sorted full list.
            # Easiest: reconstruct the full ordering once and compute a mapping name->pos.
            # To keep memory small, we’ll just re-open the tar headers once here.
        with tarfile.open(self.val_tar, "r") as tar2:
            all_members = [m for m in tar2.getmembers() if is_image_member(m)]
            all_members.sort(key=lambda m: m.name)
            pos_by_name = {m.name: i for i, m in enumerate(all_members)}

        with tarfile.open(self.val_tar, "r") as tar3:
            for m in [mm for mm in tar3.getmembers() if is_image_member(mm)]:
                # Only process the items assigned to this worker (by name)
                pos = pos_by_name[m.name]
                if (pos % num_workers) != worker_id:
                    continue

                ef = tar3.extractfile(m)
                if ef is None:
                    continue
                try:
                    with Image.open(io.BytesIO(ef.read())) as pil:
                        img = pil.convert("RGB")
                except Exception:
                    continue
                label = self.labels_0based[pos]
                yield self.transform(img), label


@dataclass
class ImageNetPaths:
    train_dir: Path  # directory containing 1000 class tar files
    val_tar: Optional[Path] = None  # path to ILSVRC2012_img_val.tar
    devkit_dir: Optional[Path] = None  # e.g., ILSVRC2012_devkit_t12/


def build_imagenet_loaders(
    paths: ImageNetPaths,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
    augment_train: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, int]]:
    """
    Returns: (train_loader, val_loader_or_None, wnid_to_idx)
    """
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    if augment_train:
        tf_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        tf_train = transforms.Compose(
            [
                transforms.Resize(256, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    tf_eval = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Build WNID mapping (prefer devkit canonical)
    if paths.devkit_dir and (paths.devkit_dir / "data" / "map_clsloc.txt").exists():
        wnid_to_idx = load_devkit_wnid_to_idx(paths.devkit_dir)
    else:
        wnid_to_idx = build_fallback_wnid_to_idx(paths.train_dir)

    ds_train = ImageNetTrainTarDataset(
        train_dir=paths.train_dir,
        transform=tf_train,
        wnid_to_idx=wnid_to_idx,
        shuffle_images_within_shard=True,
        shuffle_shards=True,
        base_seed=0,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    dl_val = None
    if paths.val_tar is not None and paths.devkit_dir is not None:
        gt_txt = paths.devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
        if not gt_txt.exists():
            raise FileNotFoundError(f"Missing validation ground truth at {gt_txt}")

        ds_val = ImageNetValTarDataset(
            val_tar=paths.val_tar,
            val_gt_txt=gt_txt,
            transform=tf_eval,
            shuffle=False,
            base_seed=0,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )
    return dl_train, dl_val, wnid_to_idx


if __name__ == "__main__":
    data_root = Path("/home/alicialu/orcd/scratch/imagenet")
    paths = ImageNetPaths(
        train_dir=data_root / "train",  # contains 1000 *.tar shards
        val_tar=data_root / "ILSVRC2012_img_val.tar",
        devkit_dir=data_root / "ILSVRC2012_devkit_t12",
    )
    batch_size = 128
    num_workers = 4
    train_loader, val_loader, wnid_to_idx = build_imagenet_loaders(
        paths,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=224,
        augment_train=True,
    )

    # quick sanity read
    it = iter(train_loader)
    imgs, labels = next(it)
    print(
        "Train batch:",
        imgs.shape,
        labels.shape,
        labels.min().item(),
        labels.max().item(),
    )

    if val_loader is not None:
        itv = iter(val_loader)
        vimgs, vlabels = next(itv)
        print(
            "Val batch:",
            vimgs.shape,
            vlabels.shape,
            vlabels.min().item(),
            vlabels.max().item(),
        )
