#!/usr/bin/env python
# coding: utf-8

# In[376]:


import numpy as np
import pandas as pd
import cvxopt
from cvxopt import matrix
import cvxpy as cp
from tqdm import tqdm
import scipy.sparse as sparse
import os
import pickle

# In[377]:


X_train = pd.read_csv("data/Xtr.csv", sep=",", index_col=0).values
Y_train = pd.read_csv("data/Ytr.csv", sep=",", index_col=0).values
X_test = pd.read_csv("data/Xte.csv", sep=",", index_col=0).values


# In[378]:


# here we have to use mistmatch kernel, 
class MismatchKernel:
    
    def __init__(self, k, m, neighbours, kmer_set, normalize=False):
        """
        kmer : chunks of words
        k : len of the kmers 
        m : number of possible mismatches
        neighbours : dictionary of all neighbours of a kmer word, 
        kmers_set: dictionary of all kmer word(set of possible word) initialize to 0
        
        """
        super().__init__()
        self.k = k
        self.m = m
        self.kmer_set = kmer_set 
        #kmer_set and neighbours have to be pre-computed (to save computational time when running multiple experiments)
        self.neighbours = neighbours
        self.normalize = normalize

    def neighbour_embed_kmer(self, x):
        """
        x: str the sequence 
        Embed kmer with neighbours.
        
        return a dict whose key are idx of kmer neigbour, composed() of the number of apparition of each kmer neigbour of kmer present inside 
        the sequence x, 
        
        
        """
        # we get all the kmer of this sequence 
        kmer_x = [x[j:j + self.k] for j in range(len(x) - self.k + 1)]
        x_emb = {}
        for kmer in kmer_x:
            neigh_kmer = self.neighbours[kmer]
            for neigh in neigh_kmer:
                idx_neigh = self.kmer_set[neigh]
                if idx_neigh in x_emb:
                    x_emb[idx_neigh] += 1
                else:
                    x_emb[idx_neigh] = 1
        return x_emb
        

    def neighbour_embed_data(self, X):
        """
        Embed data with neighbours.
        X: array of string
        """
        X_emb = []
        for i in range(len(X)):
            x = X[i]
            x_emb = self.neighbour_embed_kmer(x)
            X_emb.append(x_emb)
        return X_emb
    
    def to_sparse(self, X_emb):
        """
        Embed data to sparse matrix.
        X_emb: list of dict.
        return the corresponding matrix of this dictionary base of X_emb
        
        """
        data, row, col = [], [], []
        for i in range(len(X_emb)):
            x = X_emb[i]
            data += list(x.values())
            row += list(x.keys())
            col += [i for j in range(len(x))]
        X_sm = sparse.coo_matrix((data, (row, col)))
        return X_sm

    def gram(self, X1, X2=None):
        """ Compute the gram matrix of a data vector X where the (i,j) entry is defined as <Xi,Xj>\        X1: array of string (n_samples_1,)
        X2: array of string (n_samples_2,), if None compute the gram matrix for (X1,X1)
        """
        
        X1_emb = self.neighbour_embed_data(X1)
        X1_sm = self.to_sparse(X1_emb)
        
        if X2 is None:
            X2 = X1
        X2_emb = self.neighbour_embed_data(X2)
        X2_sm = self.to_sparse(X2_emb)

        # Reshape matrices if the sizes are different
        nadd_row = abs(X1_sm.shape[0] - X2_sm.shape[0])
        if X1_sm.shape[0] > X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row-1], [X2_sm.shape[1]-1])))
            X2_sm = sparse.vstack((X2_sm, add_row))
        elif X1_sm.shape[0] < X2_sm.shape[0]:
            add_row = sparse.coo_matrix(([0], ([nadd_row - 1], [X1_sm.shape[1] - 1])))
            X1_sm = sparse.vstack((X1_sm, add_row))

        G = (X1_sm.T * X2_sm).todense().astype('float')
        
        if self.normalize:
            G /= np.array(np.sqrt(X1_sm.power(2).sum(0)))[0,:,None]
            G /= np.array(np.sqrt(X2_sm.power(2).sum(0)))[0,None,:]
            
        return G


# In[379]:



def get_kmers(sequence, len_seq, kmer_size=3):
    """
    we split each sequence to kmer of length k
    """
    return [sequence[i: i+kmer_size] for i in range(len(sequence) - kmer_size + 1)]

def create_kmer_set(X, k, kmer_set={}):
    """
    Return a set of all kmers appearing in the dataset (Train and test dataset) with their index.
    """
    len_seq = len(X[0]) # because all the sequence have the same size, so we can just get for the first sequence
    idx = 0
    for i in range(len(X)):
        x = X[i]
        kmer_x = get_kmers(x, len_seq, k)
        for kmer in kmer_x:
            if kmer not in kmer_set: # just check if the kmer is inside the dict, and add it, if else
                kmer_set[kmer] = idx
                idx += 1
    return kmer_set


def m_neighbours(kmer, m, recurs=0):
    """
    Recursive function, who first remove m random letter from the kmer, and after remove m-1, untill 0
    for m = 1, we get an array with [kmer], and we loop over all the kmer of this array to get new array of 
    neighbour 
    that we can use in m = 2;
    Return a list of neighbours kmers (up to m mismatches).
    """
    if m == 0:
        return [kmer]

    letters = ['G', 'T', 'A', 'C']
    k = len(kmer)
    neighbours = m_neighbours(kmer, m - 1, recurs + 1)

    for j in range(len(neighbours)):
        neighbour = neighbours[j]
        for i in range(recurs, k - m + 1):
            for l in letters:
                neighbours.append(neighbour[:i] + l + neighbour[i + 1:])
    return list(set(neighbours))


def get_neighbours(kmer_set, m):
    """
    Find the neighbours given a set of kmers.
    """
    kmers_list = list(kmer_set.keys())
    kmers = np.array(list(map(list, kmers_list)))
    num_kmers, kmax = kmers.shape
    neighbours = {}
    
    # init the neigbours of kmer to empty array
    for i in range(num_kmers):
        neighbours[kmers_list[i]] = []
    
    # get the neigbours of each kmer
    for i in tqdm(range(num_kmers)):
        kmer = kmers_list[i]
        kmer_neighbours = m_neighbours(kmer, m) # get kmer neighbourd mismatched
        
        for neighbour in kmer_neighbours:
            #just check if the neighbours find exist in the kmer dict, and add it
            if neighbour in kmer_set:
                neighbours[kmer].append(neighbour)
    return neighbours

def load_or_compute_neighbors(k, m, X_train, X_test):
    """
    dataset: 0, 1 or 2\\
    k: len of the kmers
    m: number of possible mismatches
    """
    file_name = 'neighbours_k='+str(k)+'_m='+str(m)+'.p'
    full_path = 'saved_neighbors/'+file_name
    
    if os.path.exists(full_path):
        #Load the neighbors
        print('file found')
        neighbours, kmer_set = pickle.load(open(full_path, 'rb'))
    else:
        print('No file found, creating kmers neighbors')
        # because it is good to get all possible kmers from both test and train dataset
        X_full = np.hstack([X_train[:, 0], X_test[:, 0]]) 
        kmer_set = create_kmer_set(X_full, k, kmer_set={})
        
        neighbours = get_neighbours(kmer_set, m)
        pickle.dump([neighbours, kmer_set], open(full_path, 'wb'))

    return neighbours, kmer_set


# In[380]:



class KernelSVM:
    """
    SVM implementation
    
    Usage:
        svm = SVM(kernel='linear', C=1)
        svm.fit(X_train, y_train)
        svm.predict(X_test)
    """

    def __init__(self, kernel, C=1.0, tol_support_vectors=1e-4):
        """
        kernel: Which kernel to use
        C: float > 0, default=1.0, regularization parameter
        tol_support_vectors: Threshold for alpha value to consider vectors as support vectors
        """
        self.kernel = kernel
        self.C = C
        self.tol_support_vectors = tol_support_vectors

    def svm_dual_soft_to_qp_kernel(self, K, y):
        n_samples = K.shape[0]
        assert (len(y) == n_samples)
        
        P = K
        q = -y.astype('float')
        G = np.block([[np.diag(np.squeeze(y).astype('float'))],[-np.diag(np.squeeze(y).astype('float'))]])
        h = np.concatenate((self.C*np.ones(n_samples),np.zeros(n_samples)))

        return P, q, G, h

    def cvxopt_qp(self, P, q, G, h):
        P=matrix(P)
        q=matrix(q)
        G=matrix(G)
        h=matrix(h)
        solver = cvxopt.solvers.qp(P=P,q=q,G=G,h=h)
        return solver
    
    def fit_K(self, K, y):
        
        #Define the optimization problem to solve
        n_samples = K.shape[0]
        P, q, G, h = self.svm_dual_soft_to_qp_kernel(K, y)
        

        #Solve the problem
        #With cvxopt
        solver = self.cvxopt_qp(P, q, G, h)

        x = solver['x']
        self.alphas = np.squeeze(np.array(x))

        #Retrieve the support vectors
        self.support_vectors_indices = np.squeeze(np.abs(np.array(x))) > self.tol_support_vectors
        self.alphas = self.alphas[self.support_vectors_indices]
        self.support_vectors = self.X_train[self.support_vectors_indices]

        print(len(self.support_vectors), "support vectors out of",len(self.X_train), "training samples")
        return self.alphas
    
    def fit(self, X, y):
        self.X_train = X
        print("Computing the kernel...")
        G = self.kernel.gram(X)
        print("Done!")
        
        return self.fit_K(G, y)

    def predict(self, X):
        """
        X: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        K = self.kernel.gram(X, self.support_vectors)
        y = np.dot(K, self.alphas)
        return y

    def predict_classes(self, X, threshold=0):
        """
        X: array (n_samples, n_features)\\
        Return: 0 and 1 array (n_samples,)
        """
        K = self.kernel.gram(X, self.support_vectors)
        y = np.dot(K, self.alphas)
        return np.where(y > threshold, 1, -1)


# In[381]:


C = 5.0 # Parameter C for SVM
k = 12 # kmer : chunks of words, k length of kmer
m = 2 # number of possible mismatches
k_fold = 5 


# In[382]:


neighbours_0, kmer_set_0 = load_or_compute_neighbors(k,m, X_train, X_test)


# In[383]:


Y_train = np.where(Y_train == 0, -1, 1) 
# X_train = X_train[:,0]


# In[384]:



print(
    'kmer example : '+list(kmer_set_0.keys())[0],
    ', vector position '+ str(kmer_set_0[list(kmer_set_0.keys())[0]])
)
print('neighbours of this kmer : ', neighbours_0[list(neighbours_0.keys())[0]])


# In[385]:


# svm = KernelSVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_0, kmer_set=kmer_set_0,normalize=True), C=C)


# In[386]:


n = len(X_train)

seg = int(n/k_fold)
print(seg)
val_accs_0 = []
for i in range(k_fold):

    print("Doing fold", i+1,"...")
    print()

    frac_val = 1.0/k_fold
    trll = 0
    trlr = i * seg
    vall = trlr
    valr = i * seg + seg
    trrl = valr
    trrr = n
    
    train_left_indices = list(range(trll,trlr))
    train_right_indices = list(range(trrl,trrr))
    train_indices = train_left_indices + train_right_indices
    val_indices = list(range(vall,valr))
    
    svm = KernelSVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_0, kmer_set=kmer_set_0,normalize=True), C=C)
    
    sub_xtrain = X_train[train_indices]
    sub_ytrain = Y_train[train_indices]
    sub_xtest = X_train[val_indices]
    sub_ytest = Y_train[val_indices]
    
    print(sub_xtrain.shape, sub_ytrain.shape, sub_xtest.shape, sub_ytest.shape)
    svm.fit(sub_xtrain[:,0], sub_ytrain)
    
    pred_train = svm.predict_classes(sub_xtrain[:,0])
    pred_val = svm.predict_classes(sub_xtest[:,0])
    
    print(pred_train.shape, sub_ytrain.shape)
    train_acc = np.sum(np.squeeze(pred_train)==np.squeeze(sub_ytrain)) / len(sub_ytrain)
    val_acc = np.sum(np.squeeze(pred_val)==np.squeeze(sub_ytest)) / len(sub_ytest)

    print("Accuracy on train:", train_acc)
    print("Accuracy on val:", val_acc)
    val_accs_0.append(val_acc.copy())
    
print(val_accs_0)
print("Mean accuracy on val over the k folds (dataset 0):", np.mean(val_accs_0))
    


# In[387]:


[0.99, 0.985, 0.9875, 0.99, 0.9925]


# In[388]:


svm = KernelSVM(kernel=MismatchKernel(k=k, m=m, neighbours=neighbours_0, kmer_set=kmer_set_0,normalize=True), C=C)


# In[389]:


svm.fit(X_train[:,0], Y_train)


# In[390]:


svm.support_vectors.shape


# In[391]:


# X_test = X_test[:,0]
pred_0 = svm.predict_classes(X_test[:,0])
pred_0


# In[392]:


pred = np.where(pred_0 == -1, 0, 1)
pred


# In[393]:


pred = np.concatenate([pred_0.squeeze()])
pred = np.where(pred == -1, 0, 1)
pred_df = pd.DataFrame()
print(pred.shape)
print(pred)
pred_df['Covid'] = pred
pred_df.index.name = 'Id'
pred_df.index += 1
print(pred_df)
pred_df.to_csv('Yte.csv', sep=',', header=True)


# In[ ]:





# In[ ]:





# In[ ]:




