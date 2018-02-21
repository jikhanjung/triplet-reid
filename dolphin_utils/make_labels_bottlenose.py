import sys
import re
import os
import numpy as np
from prepro_utils import *


def something(data_dir, save_dir, val_split, flip_r=False, pid_mode="Bottle",
              n_queries=1, max_gallery_images=-1, stupid_phrase=""):
    dataset = []
    mkdir(save_dir)
    for path, subdirs, files in os.walk(data_dir, topdown=True):
        for filename in files:
            old_path = os.path.join(path, filename)

            # Extract pid form file path using whatever method
            pid = fid_to_pid(old_path, mode=pid_mode)
            # Add orientation to pid.
            if "(R)" in filename:
                new_pid = pid + "_R"
            else:
                new_pid = pid
            # Make new directory for this pid
            mkdir(os.path.join(save_dir, new_pid))
            new_filename = filename.replace(stupid_phrase, "")
            new_path = os.path.join(save_dir, new_pid, new_filename)
            flip = "R" in filename if flip_r else False
            move_image(os.path.join(data_dir, old_path), new_path, flip=flip)
            dataset.append([new_pid, os.path.join(new_pid, new_filename)])

    dataset = np.asarray(dataset)
    train_csv, val_csv = train_val_split1(dataset, val_split)
    np.savetxt(os.path.join(save_dir, "train.csv"), train_csv, delimiter=",", fmt="%s")
    query_csv, gallery_csv = make_val_labels(train_csv)
    # Make gallery and query csv for evaluation on train set
    np.savetxt(os.path.join(save_dir, "train_query.csv"), query_csv, delimiter=",", fmt="%s")
    np.savetxt(os.path.join(save_dir, "train_gallery.csv"), gallery_csv, delimiter=",", fmt="%s")
    if val_split > 0.:
        query_csv, gallery_csv = make_val_labels(val_csv)
        np.savetxt(os.path.join(save_dir, "val_query.csv"), query_csv, delimiter=",", fmt="%s")
        np.savetxt(os.path.join(save_dir, "val_gallery.csv"), gallery_csv, delimiter=",", fmt="%s")

    else:
        np.savetxt("train_bottlenose.csv", dataset, delimiter=",", fmt="%s")

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", metavar="PATH", required=True)
    parser.add_argument("--save_dir", metavar="PATH", required=True)
    parser.add_argument("--val_split", metavar="NUM", type=float, default=0.,
                        help="fraction of individuals to use for validation"
                             "eg 0.2-> 20% split")
    parser.add_argument("--ids_to_ignore", nargs="+", default=[],
                        help="ids of fins that shouldn't go into the dataset")
    parser.add_argument("--min_height", metavar="NUM", type=int, default=1,
                        help="Minimum height for a file to be used")
    parser.add_argument("--min_width", metavar="NUM", type=int, default=1,
                        help="Minimum width for a file to be used")
    parser.add_argument("--n_queries", metavar="NUM", type=int, default=1,
                        help="Number of query images to use in testing.")
    parser.add_argument("--max_gallery_images", metavar="NUM", type=int, default=-1,
                        help="")
    parser.add_argument("--flip_r",action="store_true",default=False,
                        help="whether to flip right facing dolphins so they face left")
    parser.add_argument("--pid_mode", choices=["Bottle", "common", "subfolder"], default="Bottle",
                        help="method for finding pid from fid")
    args = parser.parse_args(argv)
#    stupid_phrase = "(Matthew Pawley's conflicted copy 2013-12-12)"
#    stupid_regex = r"\(M.*\)"

    something(args.data_dir, args.save_dir, args.val_split, flip_r=args.flip_r, pid_mode=args.pid_mode)


if __name__ == "__main__":
    main(sys.argv[1:])