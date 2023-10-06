import torch
from torch.utils.data import DataLoader, random_split
from utils.augmentation import AugmentData


def get_train_val_test_datasets(dataset, train_ratio, val_ratio):
    assert (train_ratio + val_ratio) <= 1
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    return train_set, val_set, test_set


def get_train_val_test_loaders(dataset, train_ratio, val_ratio, train_batch_size, val_test_batch_size, num_workers):
    train_set, val_set, test_set = get_train_val_test_datasets(dataset, train_ratio, val_ratio)

    train_loader = DataLoader(train_set, train_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, val_test_batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, val_test_batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def full_data_iterator(iterable, augmentator=None):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def partial_augmentation_iterator(iterable, augmentator:AugmentData):
    iterator = iterable.__iter__()
    j=0
    while True:
        try:
            j+=1+augmentator.augmentation_count
            original_iterator = iterator.__next__()
            yield original_iterator
            for i in range(augmentator.augmentation_count):
                yield augmentator.augment_data(original_iterator, it=j*2+i)

        except StopIteration:
            iterator = iterable.__iter__()


def full_augmentation_iterator(iterable, augmentator:AugmentData):
    """
    only yields augmented/rotated data, infinite iterator
    """
    iterator = iterable.__iter__()
    j=0
    
    while True:
        try:
            j+=1
            yield augmentator.augment_data(iterator.__next__(), it=j)
            

        except StopIteration:
            iterator = iterable.__iter__()
