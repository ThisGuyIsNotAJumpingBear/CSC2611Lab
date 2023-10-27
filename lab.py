import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from gensim.models import KeyedVectors
from nltk.corpus import brown
from nltk import bigrams
from nltk.probability import FreqDist
from collections import Counter
from sklearn.decomposition import PCA
from numpy import dot
from numpy.linalg import norm

RG65 = pd.read_csv('rg65.csv')

rg_words = RG65['word0'].to_list()
rg_words.extend(RG65['word1'].to_list())
rg_words = set(rg_words)

cnt = Counter(brown.words())
common = [word[0] for word in cnt.most_common(5000)]

w = list(set(common) | rg_words)

# |W|
print('The length of W is: ', len(w))


def cos_sim(a, b):
    if norm(a)*norm(b) == 0:
        return 0
    return dot(a, b)/(norm(a)*norm(b))

def cosine_dist(a, b):
    return 1 - cos_sim(a, b)


def m1(words):
    bigram_pairs = list(bigrams(brown.words()))
    bigram_freq = FreqDist(bigram_pairs)
    cooccur_matrix = np.zeros((len(words), len(words)))
    for i in range(len(words)):
        for j in range(len(words)):
            freq = bigram_freq[(words[i], words[j])]
            cooccur_matrix[i, j] = freq
    return cooccur_matrix
    
    
def m1_plus(words):
    cooccur_matrix = m1(words)
    total_occurrences = np.sum(cooccur_matrix)
    row_sums = np.sum(cooccur_matrix, axis=1)
    col_sums = np.sum(cooccur_matrix, axis=0)
    
    ppmi_matrix = np.zeros_like(cooccur_matrix)
    
    for i in range(cooccur_matrix.shape[0]):
        for j in range(cooccur_matrix.shape[1]):
            p_x_y = cooccur_matrix[i, j] / total_occurrences
            p_x = row_sums[i] / total_occurrences
            p_y = col_sums[j] / total_occurrences
            
            pmi = np.log2(p_x_y / (p_x * p_y))
            if pmi != pmi:
                pmi = 0
            ppmi_matrix[i, j] = max(pmi, 0)
    
    return ppmi_matrix


def m2(m1_plus_matrix, num_components):
    pca = PCA(n_components=num_components)
    pca_matrix = pca.fit_transform(m1_plus_matrix)

    return pca_matrix



            
            
            
            
            
            
            
    
