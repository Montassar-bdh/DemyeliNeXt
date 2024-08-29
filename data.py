import os
import random
from collections import defaultdict
from glob import glob

import torch
from easyfsl.samplers import TaskSampler
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    AdjustContrast,
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    Spacing,
    ToTensor,
    Orientation,
)


def load_data_paths(class_paths):
    """
    Load and combine file paths for multiple classes, creating a list of tuples (data, label).

    Parameters:
    **class_paths: Keyword arguments where keys are class labels and values are paths to the class data.

    Returns:
    list: A list of tuples (file_path, label).
    """
    data_with_labels = []

    for label, path in class_paths.items():
        class_files = glob(os.path.join(path, "*"))
        data_with_labels.extend((file, label) for file in class_files)

    random.shuffle(data_with_labels)

    return data_with_labels


def split_data(data_with_labels, train_ratio=0.7, val_ratio=0.15, shuffle=True):
    """
    Split the data into training, validation, and test-MSSEG-Monastir sets based on the given ratios.
    """

    # Separate data by class
    data_by_class = defaultdict(list)
    for data, label in data_with_labels:
        data_by_class[label].append((data, label))

    train_with_data_labels = []
    val_with_data_labels = []
    test_with_data_labels = []

    # Sample from each class
    for label, items in data_by_class.items():
        class_size = len(items)
        train_size = int(train_ratio * class_size)
        val_size = int(val_ratio * class_size)
        test_size = class_size - train_size - val_size

        if shuffle:
            random.shuffle(items)

        train_with_data_labels.extend(items[:train_size])
        val_with_data_labels.extend(items[train_size : train_size + val_size])
        test_with_data_labels.extend(items[train_size + val_size :])

    return train_with_data_labels, val_with_data_labels, test_with_data_labels


def create_datasets(
    train_with_data_labels=None,
    val_with_data_labels=None,
    test_with_data_labels=None,
    train_transforms=None,
    val_transforms=None,
    test_transforms=None,
):
    # TODO: here transforms must not be none to return the datasets, change it to return dataset even transforms are none

    train_ds, val_ds, test_ds = None, None, None
    if train_with_data_labels is not None and train_transforms is not None:
        train_images, train_labels = map(list, zip(*train_with_data_labels))
        train_ds = ImageDataset(
            image_files=train_images,
            labels=train_labels,
            transform=train_transforms,
        )

    if val_with_data_labels is not None and val_transforms is not None:
        val_images, val_labels = map(list, zip(*val_with_data_labels))
        val_ds = ImageDataset(
            image_files=val_images,
            labels=val_labels,
            transform=val_transforms,
        )
    if test_with_data_labels is not None and test_transforms is not None:
        test_images, test_labels = map(list, zip(*test_with_data_labels))
        test_ds = ImageDataset(
            image_files=test_images, labels=test_labels, transform=test_transforms
        )

    return train_ds, val_ds, test_ds


def create_samplers(
    train_dataset,
    val_dataset,
    test_dataset,
    n_way=2,
    n_shot=5,
    n_query=3,
    n_train_tasks=1000,
    n_validation_tasks=150,
    n_test_tasks=1000,
):
    train_sampler = TaskSampler(
        train_dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_train_tasks,
    )

    val_sampler = TaskSampler(
        val_dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_validation_tasks,
    )

    test_sampler = TaskSampler(
        test_dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_test_tasks,
    )
    return train_sampler, val_sampler, test_sampler


def create_one_sampler(dataset, n_way=2, n_shot=5, n_query=3, n_tasks=1000):
    sampler = TaskSampler(
        dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks
    )
    return sampler


def create_transforms():
    train_transforms = Compose(
        [
            # LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="IPL"),
            Spacing(pixdim=(1.0, 1.0, 1.0)),
            Resize((32, 64, 64)),
            AdjustContrast(2.5),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            # LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="IPL"),
            Spacing(pixdim=(1.0, 1.0, 1.0)),
            Resize((32, 64, 64)),
            AdjustContrast(2.5),
            ToTensor(),
        ]
    )

    test_transforms = Compose(
        [
            # LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="IPL"),
            Spacing(pixdim=(1.0, 1.0, 1.0)),
            Resize((32, 64, 64)),
            AdjustContrast(2.5),
            ToTensor(),
        ]
    )

    return train_transforms, val_transforms, test_transforms


def create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset,
    enable_meta_loaders=True,
    enable_classic_training=True,
    n_way=2,
    n_shot=5,
    n_query=3,
    n_train_tasks=1000,
    n_validation_tasks=150,
    n_test_tasks=1000,
    num_workers=0,
    batch_size=128,
):
    """
    Create PyTorch DataLoaders for training, validation, and test datasets.
    """
    if enable_meta_loaders:
        train_dataset.get_labels = lambda: [label for label in train_dataset.labels]
        valid_dataset.get_labels = lambda: [label for label in valid_dataset.labels]
        test_dataset.get_labels = lambda: [label for label in test_dataset.labels]

        train_sampler, val_sampler, test_sampler = create_samplers(
            train_dataset,
            valid_dataset,
            test_dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_train_tasks=n_train_tasks,
            n_validation_tasks=n_validation_tasks,
            n_test_tasks=n_test_tasks,
        )

        if enable_classic_training:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=train_sampler.episodic_collate_fn,
            )

        dataloaders = {
            "train": train_loader,
            "valid": DataLoader(
                valid_dataset,
                batch_sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=False,
                collate_fn=val_sampler.episodic_collate_fn,
            ),
            "test": DataLoader(
                test_dataset,
                batch_sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=False,
                collate_fn=test_sampler.episodic_collate_fn,
            ),
        }

        return dataloaders

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }
    return dataloaders


def split_and_prepare_dataset(
    class_paths,
    n_way=2,
    n_shot=5,
    n_query=3,
    n_train_tasks=1000,
    n_validation_tasks=150,
    n_test_tasks=1000,
    batch_size=128,
    meta_dataset=True,
    enable_classic_training=False,
    shuffle=True,
):
    data_with_labels = load_data_paths(class_paths)
    train_with_data_labels, val_with_data_labels, test_with_data_labels = split_data(
        data_with_labels
    )

    if shuffle:
        random.shuffle(train_with_data_labels)
        random.shuffle(val_with_data_labels)
        random.shuffle(test_with_data_labels)

    train_transforms, val_transforms, test_transforms = create_transforms()

    train_ds, val_ds, test_ds = create_datasets(
        train_with_data_labels,
        val_with_data_labels,
        test_with_data_labels,
        train_transforms,
        val_transforms,
        test_transforms,
    )

    dataloaders = create_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        enable_meta_loaders=meta_dataset,
        enable_classic_training=enable_classic_training,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_train_tasks=n_train_tasks,
        n_validation_tasks=n_validation_tasks,
        n_test_tasks=n_test_tasks,
        num_workers=0,
        batch_size=batch_size,
    )
    return dataloaders


def create_dataset_and_dataloaders(
    split_paths,
    n_way=2,
    n_shot=5,
    n_query=3,
    n_train_tasks=1000,
    n_validation_tasks=150,
    n_test_tasks=1000,
    batch_size=128,
    meta_dataset=True,
    enable_classic_training=False,
    shuffle=True,
):
    train_with_data_labels = load_data_paths(split_paths["train"])
    val_with_data_labels = load_data_paths(split_paths["valid"])
    test_with_data_labels = load_data_paths(split_paths["test"])

    if shuffle:
        random.shuffle(train_with_data_labels)
        random.shuffle(val_with_data_labels)
        random.shuffle(test_with_data_labels)

    train_transforms, val_transforms, test_transforms = create_transforms()

    train_ds, val_ds, test_ds = create_datasets(
        train_with_data_labels,
        val_with_data_labels,
        test_with_data_labels,
        train_transforms,
        val_transforms,
        test_transforms,
    )

    dataloaders = create_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        enable_meta_loaders=meta_dataset,
        enable_classic_training=enable_classic_training,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_train_tasks=n_train_tasks,
        n_validation_tasks=n_validation_tasks,
        n_test_tasks=n_test_tasks,
        num_workers=0,
        batch_size=batch_size,
    )
    return dataloaders


def create_test_dataloader(
    class_paths,
    n_way=2,
    n_shot=5,
    n_query=3,
    n_test_tasks=100,
    num_workers=0,
    batch_size=128,
    meta_dataset=True,
    shuffle=True,
):
    test_with_data_labels = load_data_paths(class_paths)

    if shuffle:
        random.shuffle(test_with_data_labels)

    train_transforms, valid_transforms, test_transforms = create_transforms()

    train_ds, val_ds, test_ds = create_datasets(
        test_with_data_labels=test_with_data_labels, test_transforms=test_transforms
    )
    if meta_dataset:
        test_ds.get_labels = lambda: [label for label in test_ds.labels]
        test_sampler = create_one_sampler(
            test_ds, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_test_tasks
        )

        dataloader = DataLoader(
            test_ds,
            batch_sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=test_sampler.episodic_collate_fn,
        )
    else:
        dataloader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,
        )
    return dataloader
