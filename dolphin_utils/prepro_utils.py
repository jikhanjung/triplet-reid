# Just some useful stuff for preprocessing dolphin images
import os
import datetime
import numpy as np
import warnings
import pdb
from PIL import Image


def fid_to_pid(fid, mode="common"):
    if mode == "common":
        # First 4 characters of filename determines pid
        return os.path.split(fid)[-1][0:4]
    elif mode == "Bottle":
        # First 5 characters of filename determines pid
        return fid[0:5]
    elif mode == "subfolder":
        # Subfolder determines pid
        head, fn = os.path.split(fid)
        subfolder = os.path.split(head)[-1]
        return subfolder
    else:
        raise NotImplementedError("Mode {} is not implemented".format(mode))


def is_image(filename):
    return filename.lower()[-4:] in [".jpg", ".png"]


def move_image(filename, outfile, flip=False, target_size=None):
    """

    :param filename:
    :param outfile:
    :param flip:
    :param target_size:
    :return:
    """
    img = Image.open(filename)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if target_size is not None:
        img = img.resize(target_size)
    img.save(outfile)


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def make_date(filename):
    """
    (common only)
    Converts a filename from the common dolphin dataset to the date the image was taken at.
    :param filename:
    :return:
    """
    date = filename[8:14]
    year = int("20" + date[0:2])
    month = int(date[2:4])
    day = int(date[4:6])
    return datetime.date(year=year, month=month, day=day)


def rand_split(array, n, nmax):
    """
    Splits array along 0th dimension

    :param array: array to be split into query/gallery
    :param n: Number of query images
    :param nmax: Maximum number of gallery images
    :return:
    """
    array = np.asarray(array)
    permuted_array = np.random.permutation(array)
    if nmax is None or nmax < -1:
        return permuted_array[:n], permuted_array[n:]
    else:
        nmax = min(nmax, len(array))
        return permuted_array[:n], permuted_array[n:nmax]


def make_val_labels(csv_array, n_queries=1, max_gallery_examples=None):
    """
    Converts a numpy array represnting a csv of pid/fid
    :param csv_array:
    :param n_queries:
    :param max_gallery_examples:
    :return:
    """
    queries = []
    gallery = []

    pids = np.unique(csv_array[:, 0]) # Unique PIDs are in first column
    pids.sort()
    for pid in pids:
        indices_with_pid = np.where(csv_array[:,0] == pid)
        fids_with_pid = csv_array[indices_with_pid][:, 1]
        if len(fids_with_pid) <= n_queries:
            warnings.warn(
                "Couldn't select {} query image(s) from dolphin {} because it only has {} images"
                    .format(n_queries, pid, len(fids_with_pid)))
            print("I'm continuing without including it.")
            continue

        # Split all the csv rows into query and galley
        q_fids, g_fids = rand_split(fids_with_pid, n_queries, max_gallery_examples)

        queries += [[pid, fid] for fid in q_fids]
        gallery += [[pid, fid] for fid in g_fids]

    return np.asarray(queries), np.asarray(gallery)


def merge_csv_arrays(a, b):
    for pid_a in a[:, 0]:
        p
        if pid_a in b[:, 0]:
            raise Exception("amurica run no dun dun bruh")
    return np.vstack(a, b)


def train_val_split1(csv_array, val_split, split_mode="pid"):
    """
    Splits csv_array based on either the pid or fid
    :param csv_array: ndarray of [pid, fid], shape N x 2
    :param val_split: float in [0,1], fraction of images to use for validation
    :param split_mode: in ["fid", "pid"], whether to split by class or by image
    :return: train_Csv, val_csv
    """
    if split_mode == "fid":
        n = int(val_split * csv_array.shape[0])
        val_csv, train_csv = rand_split(csv_array, n, -1)
    elif split_mode == "pid":
        pids = np.unique(csv_array[:, 0]) # Unique PIDs are in first column
        pids.sort()
        n = int(val_split * len(pids))
        val_pids, train_pids = rand_split(pids, n, -1)
        pdb.set_trace()
        train_csv = np.asarray([row for row in csv_array if row[0] in train_pids])
        val_csv = np.asarray([row for row in csv_array if row[0] in val_pids])

    return train_csv, val_csv