import numpy as np
import random
import matplotlib.pyplot as plt
import warnings 
from matplotlib.animation import FuncAnimation
warnings.filterwarnings('ignore') 


class kmeans_clustering:
    '''
    Class for kmeans clustering 
    
    '''
    def __init__(self, n_clusters):
        """
        Initializes the kmeans_clustering class with the number of clusters.

        Parameters:
        n_clusters (int): The number of clusters to form.
        """
        self.n_clusters = n_clusters
        self.centroids = None
        self.centroid_history = []
        self.epoch_error_data = []
        self.iteration_error_data = []

    def plot(self, x):
        """
        Displays an image.

        Parameters:
        x (numpy array): The image to display.
        """

        plt.imshow(x)
        plt.gray()
        plt.show()

    def get_distance(self, x1, x2):
        '''
        Returns euclidean distance between two data

        Parameters:
        x1 : Image1
        x2 : Image2
        '''
        return np.sqrt(np.sum(np.power((x1 - x2), 2).flatten()))


    def plot_centroids(self):
        """
        Displays all the centroids image.

        """

        cluster_size = len(self.centroids)
        num_rows = cluster_size // 5 + int((cluster_size % 5 != 0))
        num_cols = 5
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3), squeeze=False)
        row, col = 0, 0
        msg = ['First','Second','Third','Fourth','Fifth','Sixth','Seventh','Eighth','Ninth','Tenth']
        for i in range(cluster_size):
            axs[row][col].imshow(self.centroids[i])
            axs[row][col].set_title(f'{msg[i]} Cluster')
            axs[row][col].axis('off')
            col += 1
            if col == num_cols:
                col = 0
                row += 1

        plt.tight_layout()
        plt.gray()
        plt.show()


    def plot_data(self,data):
        """
        Displays image data in a grid format
        
        """
        data_size = len(data)
        num_rows = data_size // 5 + int((data_size % 5 != 0))
        num_cols = 5
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3), squeeze=False)
        row, col = 0, 0   
        for i in range(data_size):
            axs[row][col].imshow(data[i])
            axs[row][col].set_title(f'{i} Data')
            axs[row][col].axis('off')
            col += 1
            if col == num_cols:
                col = 0
                row += 1

        plt.tight_layout()
        plt.gray()
        plt.show()

    def random_centroids(self, X):
        '''
        Randomly initializes the centroids
        To make it converge faster , we take average of random data and initialize it to centroid 
        
        Parameters
        X Numpy image data.


        Returns :
        Updated Centroids (Numpy array)

        '''
        random_image = []
        for i in range(self.n_clusters):   
            random_indices = random.sample(range(X.shape[0]), 50)
            random_image.append(np.mean(X[random_indices],axis=0))
        random_centroids = np.array(random_image)
        self.plot_data(random_centroids)

        return random_centroids 



    def batch_generator(self, data,batch_size):
        """
        Randomly returns sample data form the image data
        
        Parameters:
        data (Numpy array) : Image data
        batch_size (int) : size of batch
        """
        batch_size = min(batch_size, data.shape[0])
        random_indices = random.sample(range(data.shape[0]), batch_size)
        return data[random_indices]

    def update_label(self, sample_data, sample_label):
        """
        Updates the label of the data
        
        Parameters:
        sample_data (Numpy array) : Image data
        sample_label (list ): image labels 

        Returns :
        updated labels(list) , distance_error(float)

        """
        distance_error = 0
        for i, data in enumerate(sample_data):
            min_dist = float('inf')
            min_index = -1
            for j, centroid in enumerate(self.centroids):
                dist = self.get_distance(data, centroid)
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            sample_label[i] = min_index
            distance_error += min_dist
        distance_error /= sample_data.shape[0]
        return sample_label,distance_error

    def update_centroids(self, sample_data, sample_label):
        """
        Updates the centroids
        
        Parameters:
        sample_data (Numpy array) : Image data
        sample_label (list ): image labels 


        Returns :
        Updated Centroids (Numpy array)
        """
        temp_centroids = []
        for j in range(self.n_clusters):
            target_index = []
            for i,label in enumerate(sample_label):
                if label == j :
                    target_index.append(i)

            if target_index:
                centroid = np.mean(sample_data[target_index], axis=0)
            else:
                centroid = self.centroids[j]  # No change if no data points are assigned to this cluster
            temp_centroids.append(centroid)
        return np.asarray(temp_centroids)

    def plot_epochs(self):
        '''
        Plots epochs vs euclidean error
        '''
        plt.figure(figsize= (10,5)) 
        plt.plot(self.epoch_error_data)
        plt.xlabel("Epochs")
        plt.ylabel("Euclidean error")
        plt.title("Euclidean error vs Epoch")
        plt.show()
    
    def plot_iterations(self):
        '''
        Plots iterations vs euclidean error
        '''
        plt.figure(figsize= (10,5)) 
        plt.plot(self.iteration_error_data)
        plt.xlabel("Iterations")
        plt.ylabel("Euclidean Error")
        plt.title("Euclidean error vs Iterations")
        plt.show()

    def predict(self, data):
        '''
        Predicts which cluster the given data is close to 

        Parameters:
        data : test data

        Returns:
        Predictions (numpy array)
        '''
        predictions = []
        for data_point in data:
            min_dist = float('inf')
            min_index = -1
            for j, centroid in enumerate(self.centroids):
                dist = self.get_distance(data_point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            predictions.append(min_index)
        return predictions

    def train(self, X, batch_size=500, iterations=50, epochs=10):
        """
        Training for learning centroid weights
        
        Parameters:
        X (numpy array): The dataset.
        batch_size (int): The size of each batch.
        iterations (int): The number of iterations per epoch.
        epochs (int): The number of epochs to train.
        trained_data (bool): Whether to return the trained data.
 
        """
 
        self.centroids = self.random_centroids(X)
        self.centroid_history = []
        self.centroid_history.append(self.centroids)
        for epoch in range(epochs):
            epoch_error = 0
            print(f"Epoch: {epoch}\n---------------\n")
            new_data = self.batch_generator(X,batch_size)
            new_data_label = np.zeros(new_data.shape[0])
            for _ in range(iterations):
                new_data_label,iteration_error = self.update_label(new_data, new_data_label)
                self.centroids = self.update_centroids(new_data, new_data_label)
                self.centroid_history.append(self.centroids.copy())
                self.iteration_error_data.append(iteration_error)
                epoch_error += iteration_error
            epoch_error /= iterations
            print("Total euclidean distance error : {}".format(epoch_error))
            self.epoch_error_data.append(epoch_error)

 
    
    def train_animation(self, path):
        cluster_size = len(self.centroids)
        num_rows = cluster_size // 5 + int((cluster_size % 5 != 0))
        num_cols = 5
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4), squeeze=False)
        msg = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth', 'Tenth']
        print('\n PLEASE WAIT GENERATING ANIMATION \n')

        def update(j):
            for ax in axs.flat:
                ax.clear()
            fig.suptitle(f'Centroid Visualization: Iteration {j}')
            row, col = 0, 0
            for i in range(cluster_size):
                axs[row][col].imshow(self.centroid_history[j][i])
                axs[row][col].set_title(f'{msg[i]} Cluster')
                axs[row][col].axis('off')
                col += 1
                if col == num_cols:
                    col = 0
                    row += 1

        ani = FuncAnimation(fig, update, frames=len(self.centroid_history), interval= 200)
        plt.tight_layout()
        ani.save(filename=path, writer="pillow")
        plt.gray()
        plt.show()

    def plot_predictions(self,x_test):
        # Set the figure size
        fig = plt.figure(figsize=(10, 6))
        # Creating nine subfigures
        sfigs = fig.subfigures(3, 3)
        for i in range(3):
            for j in range(3):
                # Create subplots within each subfigure
                axs = sfigs[i][j].subplots(1, 2, squeeze=False)
                random_number = random.randint(0, x_test.shape[0] - 1)
                # Display the test image
                axs[0][0].imshow(x_test[random_number], cmap='gray')
                centroid_index = self.predict([x_test[random_number]]) 
                axs[0][0].set_title('Test Image')
                # Display the predicted centroid
                axs[0][1].imshow(self.centroids[centroid_index][0], cmap='gray')
                axs[0][1].set_title('Predicted Centroid')

        plt.show()

 



 

 
