from util import load_data
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import util

X = load_data('./image_retrieval_repository_data.pkl')
X = X[:, 1:]
pca = PCA(n_components=50)
pca.fit(X)
X_pca = pca.transform(X)
util.save_data('pca.pkl', pca)
knn_pca = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='ball_tree')
knn_pca.fit(X_pca)
util.save_data('knn.pkl', knn_pca)
