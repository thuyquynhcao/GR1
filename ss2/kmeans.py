import numpy as np
from collections import defaultdict
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Cluster:
	def __init__(self):
		self._centroid = None
		self._members = []

	def reset_members(self):
		self._members = []

	def add_member(self, member):
		self._members.append(member)


class Member:
	def __init__(self, r_d, label=None, doc_id=None):
		self._r_d = r_d
		self._label = label
		self._doc_id = doc_id


class Kmeans:
	# Hàm khởi tạo
	def __init__(self, num_clusters):
		self._num_clusters = num_clusters
		self._clusters = [Cluster() for _ in range(self._num_clusters)]
		self._E = []	# Danh sách tâm cụm centroids
		self._S = 0		# Sự tương đồng tổng thể (overall similarity)


	# Đọc dữ liệu
	def load_data(self, data_path):
		def sparse_to_dense(sparse_r_d, vocab_size):
			r_d = [0.0 for _ in range(vocab_size)]
			indices_tfidfs = sparse_r_d.split()
			for index_tfidf in indices_tfidfs:
				index = int(index_tfidf.split(':')[0])
				tfidf = float(index_tfidf.split(':')[1])
				r_d[index] = tfidf
			return np.array(r_d)

		with open(data_path) as f:
			d_lines = f.read().splitlines()
		with open('../datasets/20news-bydate/words_idfs.txt') as f:
			vocab_size = len(f.read().splitlines())

		self._data = []
		self._label_count = defaultdict(int)
		for data_id, d in enumerate(d_lines):
			features = d.split('<fff>')
			label, doc_id = int(features[0]), int(features[1])
			self._label_count[label] += 1
			r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)

			self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))


	def random_init(self, seed_value):
		np.random.seed(seed_value)
		for cluster in self._clusters:
			cluster._centroid = np.array(np.random.choice(self._data)._r_d)

	def compute_similarity(self, member, centroid):
		dis = distance.euclidean(member._r_d, centroid)
		return 1/(dis + 0.00000001)

	# Xác định cụm cho từng điểm dữ liệu
	def select_cluster_for(self, member):
		best_fit_cluster = None
		max_similarity = -1
		for cluster in self._clusters:
			similarity = self.compute_similarity(member, cluster._centroid)
			if similarity > max_similarity:
				best_fit_cluster = cluster
				max_similarity = similarity

		best_fit_cluster.add_member(member)
		return max_similarity

	# Cập nhật lại tâm cụm
	def update_centroid_of(self, cluster):
		if not cluster._members:
			return

		member_r_ds = [member._r_d for member in cluster._members]
		aver_r_d = np.mean(member_r_ds, axis=0)
		sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
		new_centroid = np.array([value / sqrt_sum_sqr for value in aver_r_d])

		cluster._centroid = new_centroid

	# Kiểm tra điều kiện dừng
	def stopping_condition(self, criterion, threshold):
		criteria = ['centroid', 'similarity', 'max_iters']
		assert criterion in criteria
		if criterion == 'max_iters':
			if self._iteration >= threshold:
				return True
			else:
				return False
		elif criterion == 'centroid':
			E_new = [list(cluster._centroid) for cluster in self._clusters]
			E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
			self._E = E_new
			if len(E_new_minus_E) <= threshold:
				return True
			else:
				return False
		else:
			new_S_minus_S = self._new_S - self._S
			self._S = self._new_S
			if new_S_minus_S <= threshold:
				return True
			else:
				return False

	def run(self, seed_value, criterion, threshold):
		self.random_init(seed_value)

		# Cập nhật liên tục các cụm cho đến khi hội tụ
		self._iteration = 0
		while True:
			# Thiết lập lại cụm, chỉ giữ lại tâm cụm centroids
			for cluster in self._clusters:
				cluster.reset_members()
			self._new_S = 0
			for member in self._data:
				max_s = self.select_cluster_for(member)
				self._new_S += max_s
			for cluster in self._clusters:
				self.update_centroid_of(cluster)

			self._iteration += 1
			if self.stopping_condition(criterion, threshold):
				break


	# Tính purity
	def compute_purity(self):
		majority_sum = 0
		for cluster in self._clusters:
			member_labels = [member._label for member in cluster._members]
			max_count = max([member_labels.count(label) for label in range(20)])
			majority_sum += max_count
		return majority_sum * 1. / len(self._data)

	# Tính NMI
	def compute_NMI(self):
		I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
		for cluster in self._clusters:
			wk = len(cluster._members) * 1.
			H_omega += - wk / N * np.log10(wk / N)
			member_labels = [member._label for member in cluster._members]
			for label in range(20):
				wk_cj = member_labels.count(label) * 1.
				cj = self._label_count[label]
				I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
		for label in range(20):
			cj = self._label_count[label] * 1.
			H_C += - cj / N * np.log10(cj / N)
		return I_value * 2. / (H_omega + H_C)


def load_data(data_path):
	def sparse_to_dense(sparse_r_d, vocab_size):
		r_d = [0.0 for _ in range(vocab_size)]
		indices_tfidfs = sparse_r_d.split()
		for index_tfidf in indices_tfidfs:
			index = int(index_tfidf.split(':')[0])
			tfidf = float(index_tfidf.split(':')[1])
			r_d[index] = tfidf
		return np.array(r_d)

	with open(data_path) as f:
		d_lines = f.read().splitlines()
	with open('../datasets/20news-bydate/words_idfs.txt') as f:
		vocab_size = len(f.read().splitlines())

	data = []
	labels = []
	label_count = defaultdict(int)
	for data_id, d in enumerate(d_lines):
		features = d.split('<fff>')
		label, doc_id = int(features[0]), int(features[1])
		label_count[label] += 1
		r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)

		data.append(r_d)
		labels.append(label)
	return np.array(data), np.array(labels)


# K-Means
def clustering_with_KMeans():
	data, _ = load_data(data_path='../datasets/20news-bydate/20news-full-tfidf.txt')
	# Sử dụng csr_matrix để tạo một ma trận thưa thớt (sparse matrix) với việc cắt hàng hiệu quả
	from sklearn.cluster import KMeans
	from scipy.sparse import csr_matrix
	X = csr_matrix(data)
	kmeans = KMeans(
		n_clusters=20,
		init='random',
		n_init=5,	# Số lần mà K-Means chạy với các tâm cụm centroids được khởi tạo khác nhau
		tol=1e-3,	# Ngưỡng (threshold) giảm lỗi tối thiểu chấp nhận được
		random_state=2018	# Thiết lập để có được kết quả xác định
	).fit(X)

	labels = kmeans.labels_

	# Đồ thị phân cụm K-Means
	max_label = max(labels)
	max_items = np.random.choice(range(X.shape[0]), size=1000, replace=False)

	pca = PCA(n_components=2).fit_transform(X[max_items, :].todense())
	tsne = TSNE().fit_transform(PCA(n_components=50).fit_transform(X[max_items, :].todense()))

	index = np.random.choice(range(pca.shape[0]), size=1000, replace=False)
	label_subset = labels[max_items]
	label_subset = [cm.hsv(i / max_label) for i in label_subset[index]]

	f, ax = plt.subplots(1, 2)

	ax[0].scatter(pca[index, 0], pca[index, 1], c=label_subset)
	ax[0].set_title('PCA Cluster')

	ax[1].scatter(tsne[index, 0], tsne[index, 1], c=label_subset)
	ax[1].set_title('TSNE Cluster')

	plt.show()

def purity_and_NMI_by_number_clusters():
	n_clusters_values = list(range(2, 21))
	purity_NMI = []
	for i in n_clusters_values:
		kmeans = Kmeans(num_clusters=i)
		kmeans.load_data(data_path='../datasets/20news-bydate/20news-full-tfidf.txt')
		kmeans.run(seed_value=2018, criterion='similarity', threshold=1000)

		purity = kmeans.compute_purity()
		NMI = kmeans.compute_NMI()
		print('Number of clusters:', i)
		print('Purity:', purity)
		print('NMI:', NMI)
		print('==============================')
		purity_NMI.append([purity, NMI])

	# Vẽ đồ thị purity và NMI ảnh hưởng bởi số lượng cluster
	plt.plot(n_clusters_values, purity_NMI)
	plt.title('Effect of number_clusters')
	plt.xlabel('Number clusters')
	plt.ylabel('Purity and NMI')
	plt.legend(loc='upper left', bbox_to_anchor=(1, 1), labels=['Purity', 'NMI'])
	plt.subplots_adjust(left=0.12, bottom=0.12, right=0.85, top=0.9)
	plt.show()


# Linear SVMs
def classifying_with_linear_SVMs():
	def compute_accuracy(predicted_y, expected_y):
		matches = np.equal(predicted_y, expected_y)
		accuracy = np.sum(matches.astype(float)) * 100. / len(expected_y)
		return accuracy

	train_X, train_y = load_data(data_path='../datasets/20news-bydate/20news-train-tfidf.txt')
	from sklearn.svm import LinearSVC
	classifier = LinearSVC(
		C=10.0,			# Hệ số phạt (penalty coefficient)
		tol=0.001,		# Dung sai (tolerance) cho tiêu chí dừng
		verbose=False	# Có in ra log (nhật ký) hay không
	)
	classifier.fit(train_X, train_y)

	test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-tfidf.txt')
	predicted_y = classifier.predict(test_X)
	accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
	print('Accuracy:', accuracy)

def linear_SVMs_accuracy_by_C():
	def compute_accuracy(predicted_y, expected_y):
		matches = np.equal(predicted_y, expected_y)
		accuracy = np.sum(matches.astype(float)) * 100. / len(expected_y)
		return accuracy

	train_X, train_y = load_data(data_path='../datasets/20news-bydate/20news-train-tfidf.txt')
	test_X, test_y = load_data(data_path='../datasets/20news-bydate/20news-test-tfidf.txt')

	from sklearn.svm import LinearSVC
	C_values = list(np.arange(1, 20.5, 0.5))
	accuracy_list = []

	for C in C_values:
		classifier = LinearSVC(
			C=float(C),		# Hệ số phạt (penalty coefficient)
			tol=0.001,		# Dung sai (tolerance) cho tiêu chí dừng
			verbose=False	# Có in ra log (nhật ký) hay không
		)
		classifier.fit(train_X, train_y)

		predicted_y = classifier.predict(test_X)
		accuracy = compute_accuracy(predicted_y=predicted_y, expected_y=test_y)
		accuracy_list.append(accuracy)

	# Vẽ đồ thị accuracy ảnh hưởng bởi C
	plt.plot(C_values, accuracy_list)
	plt.title('Effect of C')
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.show()


if __name__ == '__main__':
	clustering_with_KMeans()
	purity_and_NMI_by_number_clusters()

	classifying_with_linear_SVMs()
	linear_SVMs_accuracy_by_C()