import MinkowskiEngine as ME
import torch
import numpy as np

def collation_fn_dataset(data_labels):
    coords, feats, labels, selected_idx, pcd_indexes = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    return bcoords, feats, labels, selected_idx, pcd_indexes

def collation_fn_restricted_dataset(data_labels):
    coords, feats, labels, selected_idx, mapped_labels, pcd_indexes = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    mapped_labels = torch.from_numpy(np.concatenate(mapped_labels, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    return bcoords, feats, labels, selected_idx, mapped_labels, pcd_indexes

def collation_fn_restricted_dataset_two_samples(data_labels):
    coords, feats, labels, selected_idx, mapped_labels, coords1, feats1, labels1, selected_idx1, mapped_labels1, pcd_indexes = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoords = ME.utils.batched_coordinates(coords)
    bcoords1 = ME.utils.batched_coordinates(coords1)

    # Concatenate all lists
    feats = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels = torch.from_numpy(np.concatenate(labels, 0)).int()
    selected_idx = torch.from_numpy(np.concatenate(selected_idx, 0)).long()
    mapped_labels = torch.from_numpy(np.concatenate(mapped_labels, 0)).int()
    feats1 = torch.from_numpy(np.concatenate(feats1, 0)).float()
    labels1 = torch.from_numpy(np.concatenate(labels1, 0)).int()
    selected_idx1 = torch.from_numpy(np.concatenate(selected_idx1, 0)).long()
    mapped_labels1 = torch.from_numpy(np.concatenate(mapped_labels1, 0)).int()
    pcd_indexes = torch.tensor(pcd_indexes, dtype=torch.int16)

    return bcoords, feats, labels, selected_idx, mapped_labels, bcoords1, feats1, labels1, selected_idx1, mapped_labels1, pcd_indexes