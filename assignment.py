from gensim.models import KeyedVectors
from lab import *
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

with open('embeddings/data.pkl', 'rb') as f:
   embedding_data = pickle.load(f)
embedded_words = embedding_data['w']
embedding_data = embedding_data['E']

# w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

# m1_matrix = m1(w)
# m1_plus_matrix = m1_plus(w)
# m2_10 = m2(m1_plus_matrix, 10)
# m2_100 = m2(m1_plus_matrix, 100)
# m2_300 = m2(m1_plus_matrix, 300)
# rg = RG65.to_numpy()

# m1_sim = []
# m1_plus_sim = []
# m2_10_sim = []
# m2_100_sim = []
# m2_300_sim = []
# word_2_vec_sim = []

# for row in rg:
#     word0 = w.index(row[0])
#     word1 = w.index(row[1])
#     m1_sim.append(cosine_dist(m1_matrix[word0], m1_matrix[word1]))
#     m1_plus_sim.append(cosine_dist(m1_plus_matrix[word0], m1_plus_matrix[word1]))
#     m2_10_sim.append(cosine_dist(m2_10[word0], m2_10[word1]))
#     m2_100_sim.append(cosine_dist(m2_100[word0], m2_100[word1]))
#     m2_300_sim.append(cosine_dist(m2_300[word0], m2_300[word1]))
#     word_2_vec_sim.append(cosine_dist(w2v[word0], w2v[word1]))

# print('=================RG65 Tests=====================')
# print('Pearson correlation for M1: ', pearsonr(m1_sim, rg[:, 2]))
# print('Pearson correlation for M1+: ', pearsonr(m1_plus_sim, rg[:, 2]))
# print('Pearson correlation for M2_10: ', pearsonr(m2_10_sim, rg[:, 2]))
# print('Pearson correlation for M2_100: ', pearsonr(m2_100_sim, rg[:, 2]))
# print('Pearson correlation for M2_300: ', pearsonr(m2_300_sim, rg[:, 2]))
# print('Pearson correlation for W2V: ', pearsonr(word_2_vec_sim, rg[:, 2]))
# print()


def analogy_test(model, word0, word1, word2):
    # maximize sim(word0-word1, word2-target)
    max_sim = -100
    word_choice = None
    if type(model) == KeyedVectors:
        vec0 = model[word0]
        vec1 = model[word1]
        vec2 = model[word2]
        for word in model.key_to_index.keys():
            if word in [word0, word1, word2]:
                continue
            vec3 = model[word]
            curr_sim = sim(vec0, vec1, vec2, vec3)
            if curr_sim > max_sim:
                max_sim = curr_sim
                word_choice = word
    else:
        vec0 = model[w.index(word0)]
        vec1 = model[w.index(word1)]
        vec2 = model[w.index(word2)]
        for word in w:
            if word in [word0, word1, word2]:
                continue
            idx = w.index(word)
            vec3 = model[idx]
            curr_sim = sim(vec0, vec1, vec2, vec3)
            if curr_sim > max_sim:
                max_sim = curr_sim
                word_choice = word
    return word_choice


def sim(word0, word1, word2, word3):
    return cos_sim((word0-word1), (word2-word3))


analogy_dataset = pd.read_csv('analogy.csv').to_numpy()
subset = [item for item in analogy_dataset if all([word in w for word in item])]

# print('=================Analogy Tests=====================')
# count = [0, 0]
# for row in tqdm(subset):
#     word0 = row[0]
#     word1 = row[1]
#     word2 = row[2]
#     word3 = row[3]
#     w2v_analogy = analogy_test(w2v, word0, word1, word2)
#     m2_analogy = analogy_test(m2_300, word0, word1, word2)
#     # print(m2_analogy, word3)
#     count[0] += m2_analogy == word3
#     count[1] += w2v_analogy == word3
# print('M2_300 Accuracy: ', count[0] / len(subset))
# print('W2V_300 Accuracy: ', count[1] / len(subset))
# print()


print('=================Diachronic LSC=====================')
def l2_dist(a, b):
    return np.linalg.norm(a-b)

def entropy(vec, epi = 0.0001):
    vec_sum = np.sum(vec) + epi
    vec_norm = vec / vec_sum
    vec_norm[vec_norm <= 0] = 1
    vec_log = np.log(vec_norm)
    return 0 - np.dot(vec_norm, vec_log)

def entropy_diff(a, b):
    a = entropy(a)
    b = entropy(b)
    return abs(a-b)
    

def top_n_words(lst, vocab, n=20):
    lst = np.array(lst)
    idx = np.argsort(lst)[-n:]
    return [vocab[i] for i in idx]

def bot_n_words(lst, vocab, n=20):
    lst = np.array(lst)
    idx = np.argsort(lst)[:n]
    return [vocab[i] for i in idx]

def find_changes(data, p1, p2):
    dists = [[], [], []]
    for row in data:
        dists[0].append(cosine_dist(row[p1], row[p2]))
        dists[1].append(l2_dist(row[p1], row[p2]))
        dists[2].append(entropy_diff(row[p1], row[p2]))
    return dists

def temporal_consistency(data, interval=3):
    all_temporal = []
    for i in range(10-interval):
        dists = find_changes(data, i, i+interval)
        all_temporal.append(dists)
    all_temporal = np.array(all_temporal).transpose(1, 0, 2)
    all_temporal = np.average(all_temporal, axis=1)
    print('Consistency')
    print(np.std(all_temporal[0]))
    print(np.std(all_temporal[1]))
    print(np.std(all_temporal[2]))

def visualize(words, w, E):
    plt.figure()
    for word in words:
        idx = w.index(word)
        vecs = E[idx]
        vecs = PCA(n_components=2).fit_transform(vecs)
        x = vecs[:, 0]
        y = vecs[:, 1]
        plt.plot(x, y, label=word)
        plt.text(x[0], y[0], 's')
        plt.text(x[-1], y[-1], 'e')
    print('displaying')
    plt.legend()
    plt.savefig('visualize.png', dpi=100)

dists = find_changes(embedding_data, 0, 9)

for dist in dists:
    print('top 20: ', top_n_words(dist, embedded_words))
    print('bot 20: ', bot_n_words(dist, embedded_words))
    print()

pearson_tab = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        pearson = pearsonr(dists[i], dists[j])
        pearson_tab[i][j] = pearson.correlation

print(pearson_tab)
temporal_consistency(embedding_data, interval=5)
top_n = top_n_words(dists[0], embedded_words, n=3)
visualize(top_n, embedded_words, embedding_data)

