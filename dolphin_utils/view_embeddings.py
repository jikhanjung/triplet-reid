import sys
import pdb
import os
from itertools import product, cycle
import h5py
import numpy as np
from sklearn import manifold
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from prepro_utils import is_image

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def run_mds(distance_matrix, n_components=2):
    print("fitting MDS...")
    # Fit MDS
    mds = manifold.MDS(
        n_components=n_components,
        max_iter=3000,
        eps=1e-9,
        dissimilarity='precomputed'
    )

    pos = mds.fit(distance_matrix).embedding_
    return pos


def mpl_plot_embedding(pos, args,  n_classes):
    dimension = pos.shape[1]
    n_samples = pos.shape[0]
    palette = sns.color_palette("hls", n_classes)

    # Iterator for different markers so differnet classes are clearly displayed.
    palette = sns.color_palette("hls", 9)
    markers = ["*", "+", "x", ">", "."]
    marker_iterator = (product(markers, palette))
    fig = plt.figure()
    if args.dimension == 2:
        ax = fig.add_subplot(111)
    elif args.dimension == 3:
        ax = Axes3D(fig)


    prev_pid = None
    for i in range(n_samples):
        pid = pids[i]
        if pid != prev_pid:
            print(prev_pid,  pid)
            marker, color = marker_iterator.next()
        if dimension == 2:
            x, y = pos[i, 0], pos[i, 1]
            ax.scatter(x, y, marker=marker, c=color)
        elif dimension == 3:
            x, y, z = pos[i]
            ax.scatter(x, y, z, c=color, marker=marker)

        prev_pid = pid

    if dimension == 2:
        fig.show()
    elif dimension == 3:
        # Animate rotating 3d axis
        angle = 0
        step = 3
        while True:
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)
            angle = (angle + 3) % 360

def generate_sprite(fids, sprite_path, base_dir, res=100):
    n_images = len(fids)
    n_images_per_side = int(np.ceil(np.sqrt(n_images)))
    n_pixels_per_side =  n_images_per_side * res
    if n_pixels_per_side >= 8192:
        raise ValueError("bruh")

    sprite = np.zeros((n_pixels_per_side, n_pixels_per_side, 3), dtype=np.uint8)

    i = 0
    for (x, y) in product(range(n_images_per_side), range(n_images_per_side)):
        if i >= n_images-1:
            break
        filename = os.path.join(base_dir, fids[i])
        curr_image = Image.open(filename)
        curr_image = curr_image.resize((res, res))
        curr_image = np.array(curr_image)
        sprite[x*res:(x+1)*res, y*res:(y+1)*res, :] = curr_image
        i += 1
        print("loaded_image {}".format(fids[i]))
    sprite_img = Image.fromarray(sprite)
    sprite_img.save(sprite_path)

def generate_tsv_metadata(pids, fids, tsvpath):
    if pids is not None:
        dataset = np.vstack([pids, fids]).T
        # Add headers
        dataset = np.vstack([np.asarray([u"pid", u"fid"]), dataset])
    else:
        dataset = np.append(np.asarray("fids"), fids)
    np.savetxt(tsvpath, dataset, delimiter="\t", fmt="%s")

def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_or_dir")
    parser.add_argument("--embeddings")
    parser.add_argument("--image_root")
    parser.add_argument("--dimension",  default=2, type=int)
    parser.add_argument("--log_dir")
    parser.add_argument("--res", type=int, default=100)
    args = parser.parse_args(argv)
    with h5py.File(args.embeddings, "r") as f:
        embeddings = np.array(f["emb"])
    pids = None
    if not os.path.isdir(args.file_or_dir):
        with open(args.file_or_dir, "r") as f:
            dataset = np.genfromtxt(f, delimiter=',', dtype='|U')
            pids, fids = dataset.T
    else:
        fids = [i for i in os.listdir(args.file_or_dir) if is_image(i)]
        fids = np.asarray(fids)

    mkdir(args.log_dir)
    sprite_path = os.path.join(args.log_dir,"sprite.png",)

    generate_tsv_metadata(pids, fids, os.path.join(args.log_dir, "metadata.tsv"))
    generate_sprite(fids, sprite_path, args.image_root, args.res)
    pdb.set_trace()

    # Tensorboard stuff

    # Make Variable to store embeddings
    embedding_tensor = tf.convert_to_tensor(embeddings)
    embedding_var = tf.Variable(embedding_tensor, name="dolphin_embeds", dtype=tf.float32)
    assign_op = tf.assign(embedding_var, embedding_tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(embedding_var.eval())
#        sess.run(embedding_var.eval())

        # saver
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(args.log_dir)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        embedding.metadata_path = "metadata.tsv"
        embedding.sprite.image_path = "sprite.png"
        embedding.sprite.single_image_dim.extend([args.res, args.res])
        projector.visualize_embeddings(summary_writer, config)


        saver.save(sess, os.path.join(args.log_dir, "embeds.cpkt"))

#    distance_matrix = euclidean_distances(embeddings)
#    pos = run_mds(distance_matrix, args.dimension)
if __name__ == "__main__":
    main(sys.argv[1:])