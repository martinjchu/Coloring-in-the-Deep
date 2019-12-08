import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D
import hyperparameters as hp
import sklearn.neighbors as nn
# from preprocess import get_train_data


def main():


    # l_images, ab_images, batch_labels = get_train_data()

    l_images = np.load('l_images.npy')
    ab_images = np.load('ab_images.npy')
    print(ab_images.shape)

    # image = np.load('img.npy')
    # print(image.shape)
    clusters = np.load('pts_in_hull.npy')
    # print(clusters.shape)

    KNN = nn.NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree').fit(clusters)

    #w * h * 2
    # abimg = image[:, :, 1:] 
    # print(abimg[0][1][0])
    # print(abimg[0][1][1])

    # ab_points = abimg.reshape(-1, 2)

    ab_pts = ab_images.reshape(-1, 2)
    print(ab_pts.shape)


    # labels = np.zeros((abimg.shape[0]*abimg.shape[1], 313))

    labels = np.zeros((ab_images.shape[0], ab_images.shape[1]*ab_images.shape[2], 313))
    print(labels.shape)

    dists, inds = KNN.kneighbors(ab_pts, 5)

    np.save('dists', dists)
    np.save('inds', inds)

    weights = np.exp(-dists**2/(2*5**2))
    weights = weights/np.sum(weights, axis = 1).reshape(-1,1)

    pixel_ind = np.arange(abimg.shape[0]*abimg.shape[1])[:, np.newaxis]
    print(pixel_ind.shape)
    labels[pixel_ind, inds] = weights
    







if __name__ == '__main__':
   main()
