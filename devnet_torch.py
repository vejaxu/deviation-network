import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import vstack, csc_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

import time
import sys
import argparse
import random

from utils_torch import *

seed = 42  
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


MAX_INT = np.iinfo(np.int32).max
data_format = 0


class DevNetD(nn.Module):
    def __init__(self, input_shape):
        super(DevNetD, self).__init__()
        self.input_shape = input_shape

        self.layer1 = nn.Linear(input_shape[0], 1000)
        self.layer2 = nn.Linear(1000, 250)
        self.layer3 = nn.Linear(250, 20)
        self.output_layer = nn.Linear(20, 1)

        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.layer1(X))
        X = self.relu(self.layer2(X))
        X = self.relu(self.layer3(X))
        X = self.output_layer(X)
        return X

class DevNetS(nn.Module):
    def __init__(self, input_shape):
        super(DevNetS, self).__init__()
        self.input_shape = input_shape

        self.layer1 = nn.Linear(input_shape[0], 20)
        self.output_layer = nn.Linear(20, 1)

        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.layer1(X))
        X = self.output_layer(X)

        return X


class DevNetLinear(nn.Module):
    def __init__(self, input_shape):
        super(DevNetLinear, self).__init__()
        self.input_shape = input_shape
        self.output_layer = nn.Linear(input_shape[0], 1) 

    def forward(self, X):
        return self.output_layer(X)


def deviation_loss(y_label, y_predict):
    confidence_margin = 5.

    ref = torch.normal(mean=torch.zeros(5000), std=torch.ones(5000), dtype=torch.float32)

    dev = (y_predict -  torch.mean(ref)) / torch.std(ref)

    inlier_loss =  torch.abs(dev)

    outlier_loss = torch.abs(torch.max(confidence_margin - dev, torch.zeros_like(dev)))

    return torch.mean((1 - y_label) * inlier_loss + y_label * outlier_loss)


def deviation_network(input_shape, network_depth):
    if network_depth == 4:
        model = DevNetD(input_shape)

    elif network_depth == 2:
        model = DevNetS(input_shape)

    elif network_depth == 1:
        model = DevNetLinear(input_shape)
    
    else:
        sys.exit("The network depth is not set properly")
    
    optimizer = optim.RMSprop(model.parameter(), lr=0.001, weight_decay=0.01)

    return model, optimizer


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:                
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0
 
def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

 
def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    '''      
    ref = np.empty((batch_size))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.array(training_labels)


def load_model_weight_predict(model_name, input_shape, network_depth, x_test, device='cpu'):
    """
    Load the saved weights and make predictions on the test data.

    Parameters:
    - model_name: Path to the saved model weights.
    - input_shape: The shape of the input data.
    - network_depth: The depth of the model (1, 2, or 4).
    - x_test: The test data (Numpy array or sparse matrix).
    - device: The device to run the model on ('cpu' or 'cuda').

    Returns:
    - scores: The predicted anomaly scores.
    """
    # Initialize model
    model = deviation_network(input_shape, network_depth).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_name, map_location=device))

    # Set model to evaluation mode (important for dropout, batch normalization, etc.)
    model.eval()

    # Convert x_test to a PyTorch tensor if it's not already
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    # Prediction logic
    if data_format == 0:
        with torch.no_grad():
            scores = model(x_test_tensor).cpu().numpy()  # Get predictions
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test_tensor[i:count].to(device)
            with torch.no_grad():
                scores[i:count] = model(subset).cpu().numpy()
            if i % 1024 == 0:
                print(f"Processed {i} samples")
            i = count
            count += 512
            if count > data_size:
                count = data_size

        assert count == data_size

    return scores


def inject_noise_sparse(seed, n_out, random_seed):  
    '''
    Add anomalies to training data to replicate anomaly-contaminated datasets.
    Randomly swap 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    '''
    rng = np.random.RandomState(random_seed)  # Initialize RNG with provided seed
    n_sample, dim = seed.shape  # Get number of samples and features
    swap_ratio = 0.05  # Define 5% swap ratio
    n_swap_feat = int(swap_ratio * dim)  # Calculate the number of features to swap

    seed = seed.tocsc()  # Convert input matrix to CSC format for efficient indexing
    noise = csc_matrix((n_out, dim))  # Initialize a sparse matrix to hold the noise (outliers)
    
    for i in np.arange(n_out):  # Loop to create n_out outliers
        outlier_idx = rng.choice(n_sample, 2, replace=False)  # Randomly select two samples for swapping features
        o1 = seed[outlier_idx[0]]  # First outlier sample
        o2 = seed[outlier_idx[1]]  # Second outlier sample
        
        swap_feats = rng.choice(dim, n_swap_feat, replace=False)  # Randomly choose features to swap
        
        noise[i] = o1.copy()  # Copy the first outlier sample into noise matrix
        noise[i, swap_feats] = o2[0, swap_feats]  # Swap the selected features from the second sample
        
    return noise.tocsr()  # Return the result as a CSR sparse matrix


def inject_noise(seed, n_out, random_seed):   
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''  
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    noise = np.empty((n_out, dim))
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[swap_feats]
    return noise


def run_devnet(args):
    names = args.dataset.split(',')
    names = ['annthyroid_21feat_normalised']
    network_depth = int(args.network_depth)
    random_seed = args.ramdn_seed
    for nm in names:
        runs = args.runs
        rauc = np.zeros(runs)
        ap = np.zeros(runs)  
        filename = nm.strip()
        global data_format
        data_format = int(args.data_format)
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            x = x.tocsr()    
        outlier_indices = np.where(labels == 1)[0]
        outliers = x[outlier_indices]  
        n_outliers_org = outliers.shape[0]   
        
        train_time = 0
        test_time = 0
        for i in np.arange(runs):  
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42, stratify = labels)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            
            n_noise  = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            n_noise = int(n_noise)                
            
            rng = np.random.RandomState(random_seed)  
            if data_format == 0:                
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)            
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)
                
                noises = inject_noise(outliers, n_noise, random_seed)
                x_train = np.append(x_train, noises, axis = 0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            else:
                if n_outliers > args.known_outliers:
                    mn = n_outliers - args.known_outliers
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)        
                    retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                    retain_idx = list(retain_idx)
                    x_train = x_train[retain_idx]
                    y_train = y_train[retain_idx]                               
                
                noises = inject_noise_sparse(outliers, n_noise, random_seed)
                x_train = vstack([x_train, noises])
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            outlier_indices = np.where(y_train == 1)[0]
            inlier_indices = np.where(y_train == 0)[0]
            print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
            input_shape = x_train.shape[1:]
            n_samples_trn = x_train.shape[0]
            n_outliers = len(outlier_indices)            
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            
            
            start_time = time.time() 
            input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size    
            nb_batch = args.nb_batch  
            model = deviation_network(input_shape, network_depth)
            print(model.summary())  
            model_name = "./model/devnet_"  + filename + "_" + str(args.cont_rate) + "cr_"  + str(args.batch_size) +"bs_" + str(args.known_outliers) + "ko_" + str(network_depth) +"d.h5"
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only = True, save_weights_only = True)            
            model.fit_generator(batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                                steps_per_epoch = nb_batch,
                                epochs = epochs,
                                callbacks=[checkpointer])   
            train_time += time.time() - start_time
            
            start_time = time.time() 
            scores = load_model_weight_predict(model_name, input_shape, network_depth, x_test)
            test_time += time.time() - start_time
            rauc[i], ap[i] = aucPerformance(scores, y_test)     
        
        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        train_time = train_time/runs
        test_time = test_time/runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))    
        print("average runtime: %.4f seconds" % (train_time + test_time))
        writeResults(filename+'_'+str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org, n_outliers,
                     network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)
        

parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1','2', '4'], default='2', help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
parser.add_argument("--runs", type=int, default=10, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=30, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
parser.add_argument("--dataset", type=str, default='annthyroid_21feat_normalised', help="a list of data set names")
parser.add_argument("--data_format", choices=['0','1'], default='0',  help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str, default='./results/devnet_auc_performance_30outliers_0.02contrate_2depth_10runs.csv', help="the output file path")
parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
args = parser.parse_args()
run_devnet(args)