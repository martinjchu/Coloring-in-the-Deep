import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D
import hyperparameters as hp
import sklearn.neighbors as nn









def main():
    image = np.load('img.npy')
    print(image.shape)
    clusters = np.load('pts_in_hull.npy')
    print(clusters.shape)

    KNN = nn.NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree').fit(clusters)

    #w * h * 2
    abimg = image[:, :, 1:] 
    # print(abimg[0][1][0])
    # print(abimg[0][1][1])
    ab_points = abimg.reshape(-1, 2)
    # print(ab_points)
    labels = np.zeros((abimg.shape[0]*abimg.shape[1], 313))

    dists, inds = KNN.kneighbors(ab_points, 5)

    weights = np.exp(-dists**2/(2*5**2))
    weights = weights/np.sum(weights, axis = 1).reshape(-1,1)

    pixel_ind = np.arange(abimg.shape[0]*abimg.shape[1])[:, np.newaxis]
    print(pixel_ind.shape)
    labels[pixel_ind, inds] = weights


    print(labels[100])




    







if __name__ == '__main__':
   main()
