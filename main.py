from keras.datasets import mnist 
import numpy as np
import kmeans
import argparse 
import warnings 
warnings.filterwarnings('ignore') 

if __name__ == "__main__" :
   
    parser = argparse.ArgumentParser() 
    parser.add_argument("--epochs",'-e', type = int , help = " \n Pass the epochs to train for")
    parser.add_argument("--iteration",'-i', type = int , help = "\n Pass the number of iteration in a epoch")
    parser.add_argument("--batchsize",'-b', type = int , help = "\n Pass the number of batch size to pass in each epochs")
    args = parser.parse_args()


    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255 # normalize
    n_cluster = 10
    clusterizer = kmeans.kmeans_clustering(n_cluster)
    clusterizer.train(x_train,batch_size = args.batchsize ,iterations = args.iteration, epochs= args.epochs)
    clusterizer.plot_centroids()
    clusterizer.plot_iterations()
    clusterizer.plot_predictions(x_test/255)
   # clusterizer.train_animation('clustering.gif')

     
