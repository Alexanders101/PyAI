import warnings
warnings.filterwarnings("ignore")

__author__ = 'alex'
__version__ = 2.0

from sys import version_info

if version_info[0] == 2:  # Python 2.x
    from utils import *
elif version_info[0] == 3:  # Python 3.x
    # from PyAI.utils import *
    from utils import *

from sklearn import *
from scipy import stats
from scipy.spatial.distance import euclidean
from time import time
from copy import copy
from collections import OrderedDict as oDict
from sklearn.preprocessing.data import binarize


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# region Models
class CLUSTER:
    __name__ = 'Clustering'
    KMean = cluster.KMeans
    MiniBatch = cluster.MiniBatchKMeans
    AffProp = cluster.AffinityPropagation
    MShift = cluster.MeanShift
    Spectral = cluster.SpectralClustering
    # Ward = cluster.Ward
    Agglomerative = cluster.AgglomerativeClustering
    DBScan = cluster.DBSCAN


class SVM:
    __name__ = 'Support Vector Machine'
    Regular = (svm.SVC, svm.SVR)
    Linear = (svm.LinearSVC, svm.LinearSVR)
    Nu = (svm.NuSVC, svm.NuSVR)


class GMM:
    __name__ = 'Gaussian Mixture'
    REGULAR = mixture.GMM
    VARIATIONAL_INFINITE = mixture.DPGMM
    VARIATIONAL = mixture.VBGMM


class LINEAR:
    BAYESIAN_RIDGE = linear_model.BayesianRidge
    RIDGE = linear_model.Ridge
    RIDGE_CLASS = linear_model.RidgeClassifier
    LOGISTIC = linear_model.LogisticRegression
    KERNEL_RIDGE = kernel_ridge.KernelRidge
    SGD = linear_model.SGDRegressor
    SGD_CLASS = linear_model.SGDClassifier

class DISCRIMINANT_ANALYSIS:
    LINEAR = lda.LDA
    QUADRATIC = qda.QDA

class ISOTONIC:
    REGRESSION = isotonic.IsotonicRegression

class NAIVE_BAYES:
    REGULAR = naive_bayes.GaussianNB
    MULTINOMIAL = naive_bayes.MultinomialNB
    BERNOULLI = naive_bayes.BernoulliNB

class SEMISUPERVISED:
    PROPAGATION = semi_supervised.LabelPropagation
    SPREADING = semi_supervised.LabelSpreading

# endregion


class TRANSFORMATION:
    @staticmethod
    def Normalize():
        return ('normalization', preprocessing.Normalizer())

    @staticmethod
    def Standardize():
        return ('standardization', preprocessing.StandardScaler())

    @staticmethod
    def MinMaxScale():
        return ('min_max', preprocessing.MinMaxScaler())

    @staticmethod
    def Binarize():
        return ('binarization', preprocessing.Binarizer())

    @staticmethod
    def GaussianProjection(nComponents='auto'):
        return ('gaussian_projection', random_projection.GaussianRandomProjection(nComponents))

    @staticmethod
    def SparseProjection(nComponents='auto'):
        return ('sparse_projection', random_projection.SparseRandomProjection(nComponents))

    @staticmethod
    def VarianceThreshold(threshold=0.0):
        return ('variance_threshold', feature_selection.VarianceThreshold(threshold))

    @staticmethod
    def Isomap(nNeighbors=5, nComponents=2):
        return ('isomap', manifold.Isomap(nNeighbors, nComponents))

    @staticmethod
    def LocalLinear(nNeighbors=5, nComponents=2):
        return ('local_linear', manifold.LocallyLinearEmbedding(nNeighbors, nComponents))

    class PCA:
        @staticmethod
        def RandomizedPCA(nComponents=None):
            return ('random_pca', decomposition.RandomizedPCA(nComponents))

        @staticmethod
        def SparsePCA(nComponents=None):
            return ('sparse_pca', decomposition.SparsePCA(nComponents))

        @staticmethod
        def LinearPCA(nComponents=None):
            return ('linear_pca', decomposition.PCA(nComponents))

        @staticmethod
        def RBFPCA(nComponents=None):
            return ('rbf_pca', decomposition.KernelPCA(nComponents, 'rbf'))

        @staticmethod
        def SigmoidPCA(nComponents=None):
            return ('sigmoid_pca', decomposition.KernelPCA(nComponents, 'sigmoid'))

        @staticmethod
        def CosinePCA(nComponents=None):
            return ('cosine_pca', decomposition.KernelPCA(nComponents, 'cosine'))

        @staticmethod
        def FastICA(nComponents=None):
            return ('fast_ica', decomposition.FastICA(nComponents))

    class ClUSTER_BASED:
        @staticmethod
        def KMeans(nDimensions=8):
            return ('KMeans', cluster.KMeans(nDimensions))

        @staticmethod
        def FeatureAgglomeration(nDimensions=2):
            return ('feature_agglomeration', cluster.FeatureAgglomeration(nDimensions))

    class KERNEL_APPROXIMATION:
        @staticmethod
        def RBF():
            return ('rbf_sampler', kernel_approximation.RBFSampler())

        @staticmethod
        def Nystroem():
            return ('nystroem', kernel_approximation.Nystroem())


class Brain:
    def __str__(self):
        if self.__classification:
            n_classes = len(unique(self.__y_labels))
        else:
            n_classes = 'N/A'
        return ("Number of Samples:        \t\t{}\n"
                "Number of Features:       \t\t{}\n"
                "Number of Classes:        \t\t{}\n"
                "Regression Data:          \t\t{}\n"
                "Classification Data:      \t\t{}\n\n"
                "Clustering Enabled:       \t\t{}\n"
                "Nearest Neighbors Enabled:\t\t{}\n"
                "SVM Enabled:              \t\t{}\n"
                "Gaussian Mixture Enabled: \t\t{}".format(self.__n_samples, self.__n_features, n_classes,
                                                          self.__regression,
                                                          self.__classification, self.__E_cluster, self.__E_knn,
                                                          self.__E_svm,
                                                          self.__E_gmm))

    def __repr__(self):
        if self.__classification:
            n_classes = len(unique(self.__y_labels))
        else:
            n_classes = 'N/A'
        return ("n_samples:      {}\n"
                "n_features:     {}\n"
                "n_classes:      {}\n"
                "regression:     {}\n"
                "classification: {}\n"
                "E_cluster:      {}\n"
                "E_knn:          {}\n"
                "E_svm:          {}\n"
                "E_gmm:          {}".format(self.__n_samples, self.__n_features, n_classes, self.__regression,
                                            self.__classification, self.__E_cluster, self.__E_knn, self.__E_svm,
                                            self.__E_gmm))

    def __init__(self, x_data, y_labels=None, y_data=None, verbose=False):
        """The base class of the PyAI library.

        This Class will handle all of the initial set up for your data, the only necessary input is your X Data.

        Attributes:
            x_data (nd_array): An array of shape (nSamples, nFeatures) that stores the data you are learning from
            y_data (nd_array, None): An array of shape nSamples that stores known values of each data point"""

        self.__n_samples, self.__n_features = x_data.shape

        # Start: Error Handling
        if self.__n_samples == 0 or self.__n_features == 0:
            print('Error! No Data')
            return
        if not check_data(x_data):
            return
        # End: Error Handling

        # Set up global data
        self.__x_data = x_data

        # Set up regression data if it exists
        self.__y_data = y_data
        self.__regression = y_data is not None

        # Set up classification data if it exists
        self.__y_labels = y_labels
        self.__classification = y_labels is not None

        # If either regression or classification data is given, the data is supervised
        self.__supervised = self.__classification or self.__regression

        self.__data_transformed = False
        # if transform is not None:
        #     self.__init_data_handling(transform)

        self.__verbose = verbose

        self.__E_cluster = False
        self.__E_est = False
        self.__E_knn = False
        self.__E_svm = False
        self.__E_gmm = False
        self.__E_neural = False
        self.__E_nb = False
        self.__E_ss = False

        self.__cur_data_transformed = False
        self.__labels_predicted = False

    def init_data_transformation(self, *transformations):
        self.__manipulator = pipeline.Pipeline(transformations)
        self.__manipulator.fit(self.__x_data)
        self.__x_data = self.__manipulator.transform(self.__x_data)
        self.__data_transformed = True
        self.refit_models()

    # Universal Get Methods
    def get_point(self, index='all'):
        if index == 'all':
            return self.__x_data
        return self.__x_data[index]

    @regression_method()
    def get_data(self, index='all'):
        if index == 'all':
            return self.__y_data
        return self.__y_data[index]

    @classification_method()
    def get_label(self, index='all'):
        if index == 'all':
            return self.__y_labels
        return self.__y_labels[index]

    """CLUSTER BEGIN"""
    # Clustering init Methods
    def init_clustering(self, model=CLUSTER.MiniBatch, **model_params):
        # Set params for various clustering types
        print("\nStarting Clustering")
        if model is CLUSTER.MiniBatch:
            n_clusters = handle_required(model_params, ['n_clusters'])
            batch_size, init = handle_optional(model_params, [['batch_size', 100],
                                                              ['init', 'k-means++']])
            self.__cluster_type = 'Mini Batch KMeans Clustering'
            self.__cluster = model(n_clusters=n_clusters, batch_size=batch_size, init=init)

        if model is CLUSTER.KMean:
            n_clusters = handle_required(model_params, ['n_clusters'])
            batch_size, init = handle_optional(model_params, [['init', 'k-means++']])

            self.__cluster_type = 'KMeans Clustering'
            self.__cluster = model(n_clusters=n_clusters, init=init)

        if model is CLUSTER.DBScan:
            eps, min_samples = handle_optional(model_params, [['eps', 0.5],
                                                              ['min_samples', 5]])
            self.__cluster_type = 'Density Based Clustering'
            self.__cluster = model(eps=eps, min_samples=min_samples)

        if model is CLUSTER.AffProp:
            damping, convergence_iter = handle_optional(model_params, [['damping', 0.5],
                                                                       ['convergence_iter', 15]])
            self.__cluster_type = 'Affinity Propagation'
            self.__cluster = model(damping=damping, convergence_iter=convergence_iter)

        if model is CLUSTER.Agglomerative:
            n_clusters = handle_required(model_params, ['n_clusters'])
            linkage = handle_optional(model_params, [['linkage', 'ward']])

            self.__cluster_type = 'Agglomerative Clustering'
            self.__cluster = model(n_clusters=n_clusters, linkage=linkage)


        # Make model with params
        t0 = time()
        self.__cluster.fit(self.__x_data)
        t1 = time()

        # make miscellaneous variables
        self.__cluster_params = (model, model_params)
        self.__cluster_labels = self.__cluster.labels_
        self.__cluster_classes = unique(self.__cluster_labels)
        self.__cluster_centers = self.__calculate_cluster_centers()
        self.__cluster_regression_centers = self.__calculate_cluster_regression_centers()
        self.__E_cluster = True

        print('Time to complete ' + self.__cluster_type + ' was:\t%.2fs' % (t1 - t0))
        if self.__verbose:
            print(self.__cluster)

    def init_estimate_labels(self, model=CLUSTER.MiniBatch, **model_params):
        if self.__classification:
            print("Labels already provided, no need to estimate.")
            return

        enabled = self.__E_cluster
        if enabled:
            cluster_hold = (self.__cluster, self.__cluster_type, self.__cluster_labels,
                            self.__cluster_classes, self.__cluster_centers, self.__cluster_regression_centers)

        self.init_clustering(model, **model_params)

        self.__y_labels = self.__cluster_labels
        self.__classification = True
        self.__E_cluster = enabled
        self.__est_params = (model, model_params)
        self.__E_est = True
        if enabled:
            self.__cluster, self.__cluster_type, self.__cluster_labels, \
            self.__cluster_classes, self.__cluster_centers, self.__cluster_regression_centers = cluster_hold
            del cluster_hold
        else:
            del self.__cluster
            del self.__cluster_type
            del self.__cluster_labels
            del self.__cluster_classes
            del self.__cluster_centers
            del self.__cluster_regression_centers

    # Clustering Utility Methods
    def __calculate_cluster_centers(self):
        try:
            return self.__cluster.cluster_centers_
        except AttributeError:
            return array([mean(self.__x_data[self.__cluster_labels == label], 0)
                          for label in unique(self.__cluster_labels)])

    def __calculate_cluster_regression_centers(self):
        if not self.__regression:
            return None
        return array([mean(self.__y_data[self.__cluster_labels == label], 0)
                      for label in unique(self.__cluster_labels)])

    def __get_distances(self, x):
        try:
            return self.__cluster.transform(x)[0]
        except AttributeError:
            return array([euclidean(x, center) for center in self.__cluster_centers])

    # Clustering Get Methods
    @init_check('_Brain__E_cluster')
    def get_cluster(self, label):
        return self.__x_data[self.__cluster_labels == label]

    @regression_method()
    @init_check('_Brain__E_cluster')
    def get_cluster_data(self, label):
        return self.__y_data[self.__cluster_labels == label]

    @init_check('_Brain__E_cluster')
    def get_cluster_labels(self):
        return self.__cluster_labels

    # Clustering Predict Methods
    @init_check('_Brain__E_cluster')
    @data_transform()
    def predict_cluster_class(self, x):
        try:
            prediction = self.__cluster.predict(x)
        except AttributeError:
            print(self.__cluster_type + ' is not able to predict points')
            return None

        if len(prediction) == 1:
            return prediction[0]
        return prediction

    @init_check('_Brain__E_cluster')
    @data_transform()
    def predict_cluster_fuzzy(self, x):
        if len(x.shape) is 1:
            x = array([x])
        data = array([transpose(array([self.__cluster_classes,
                                       weights_from_distances(self.__get_distances(point), 3, False)], object))
                      for point in x])
        if len(data) == 1:
            return data[0]
        return data

    @regression_method()
    @init_check('_Brain__E_cluster')
    @data_transform()
    def predict_cluster_data(self, x):
        if len(x.shape) is 1:
            x = array([x])
        data = array([weighted_average([self.__cluster_regression_centers,
                                        weights_from_distances(self.__get_distances(point), 3, False)])
                      for point in x])
        if len(data) == 1:
            return data[0]
        return data

    """CLUSTER END"""

    """NEAREST NEIGHBORS BEGIN"""

    def init_neighbors(self, n_neighbors=5, radius=1.0):
        """Unsupervised Method: Nearest Neighbors Initialization Method
            nNeighbors = number of neighbors for n based algorithm
            radius = radius for distance based algorithm"""
        print("\nStarting Nearest Neighbors")

        self.__knn = neighbors.NearestNeighbors(n_neighbors, radius)
        t0 = time()
        self.__knn.fit(self.__x_data)
        t1 = time()

        self.__knn_params = (n_neighbors, radius)
        self.__E_knn = True
        print('Time to complete Nearest Neighbors was:\t%.2fs' % (t1 - t0))
        if self.__verbose:
            print(self.__knn)
        return True

    # Nearest Neighbor Get Methods
    @init_check('_Brain__E_knn')
    @data_transform()
    def get_knn_nearest(self, x, n_neighbors=None, ind=False):
        indices = self.__knn.kneighbors(x, n_neighbors)[1]
        if ind:
            return indices
        return array([self.__x_data[index] for index in indices])

    @data_transform()
    @init_check('_Brain__E_knn')
    def get_knn_radius(self, x, radius=None, ind=False):
        indices = self.__knn.radius_neighbors(x, radius)[1]
        if ind:
            return indices
        return array([self.__x_data[index] for index in indices])

    # Nearest Neighbors Predict Methods
    @classification_method()
    @init_check('_Brain__E_knn')
    @data_transform()
    def predict_knn_nearest_class(self, x, weighted=False, n_neighbors=None):
        if type(x) is not ndarray:
            x = array(x)
        if len(shape(x)) == 1:
            x = x.reshape(1, -1)

        distances, indices = self.__knn.kneighbors(x, n_neighbors)

        classes_ = unique(self.__y_labels)
        y = self.__y_labels
        if y.ndim == 1:
            y = y.reshape((-1, 1))
            classes_ = [classes_]

        n_outputs = len(classes_)
        n_samples = x.shape[0]

        y_pred = empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if not weighted:
                mode, _ = stats.mode(y[indices, k], axis=1)
            else:
                weights = [weights_from_distances(distance) for distance in distances]
                mode, _ = weighted_mode(y[indices, k], weights, axis=1)

            mode = asarray(mode.ravel(), dtype=intp)
            y_pred[:, k] = classes_k.take(mode)

        if self.__y_labels.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]

        return y_pred

    @regression_method()
    @init_check('_Brain__E_knn')
    @data_transform()
    def predict_knn_nearest_data(self, x, weighted=False, n_neighbors=None):
        distances, indices = self.__knn.kneighbors(x, n_neighbors)

        y = self.__y_data
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if not weighted:
            y_pred = mean(y[indices], axis=1)
        else:
            weights = [weights_from_distances(distance) for distance in distances]
            y_pred = array([(average(y[ind, :], axis=0,
                                     weights=weights[i]))
                            for (i, ind) in enumerate(indices)])
        if self.__y_data.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]
        return y_pred

    @classification_method()
    @init_check('_Brain__E_knn')
    @data_transform()
    def predict_knn_radius_class(self, x, weighted=False, radius=None):
        if type(x) is not ndarray:
            x = array(x)
        if len(shape(x)) == 1:
            x = x.reshape(1, -1)

        n_samples = x.shape[0]

        distances, indices = self.__knn.radius_neighbors(x, radius)
        inliers = [i for i, nind in enumerate(indices) if len(nind) != 0]
        outliers = [i for i, nind in enumerate(indices) if len(nind) == 0]

        classes_ = unique(self.__y_labels)
        y = self.__y_labels
        if y.ndim == 1:
            y = y.reshape((-1, 1))
            classes_ = [classes_]
        n_outputs = len(classes_)

        if outliers:
            raise ValueError('No neighbors found for test samples %r, '
                             'you can try using larger radius, '
                             'give a label for outliers, '
                             'or consider removing them from your dataset.'
                             % outliers)

        y_pred = empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            pred_labels = array([y[ind, k] for ind in indices],
                                dtype=object)
            if not weighted:
                mode = array([stats.mode(pl)[0]
                              for pl in pred_labels[inliers]], dtype=int)
            else:
                weights = [weights_from_distances(distance) for distance in distances]
                mode = array([weighted_mode(pl, w)[0]
                              for (pl, w)
                              in zip(pred_labels[inliers], weights)],
                             dtype=int)

            mode = mode.ravel()

            y_pred[inliers, k] = classes_k.take(mode)

        if outliers:
            y_pred[outliers, :] = -1

        if self.__y_labels.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]

        return y_pred

    @regression_method()
    @init_check('_Brain__E_knn')
    @data_transform()
    def predict_knn_radius_data(self, x, weighted=False, radius=None):
        distances, indices = self.__knn.radius_neighbors(x, radius)

        y = self.__y_data
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if not weighted:
            y_pred = array([mean(y[index, :], axis=0)
                            for index in indices])
        else:
            weights = [weights_from_distances(distance) for distance in distances]
            y_pred = array([(average(y[index, :], axis=0,
                                     weights=weights[i]))
                            for (i, index) in enumerate(indices)])
        if self.__y_data.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]

        return y_pred

    """NEAREST NEIGHBORS END"""

    """SVM START"""

    def init_svm(self, model=SVM.Regular, **model_params):

        if (not self.__classification) and (not self.__regression):
            raise AttributeError('SVM is a supervised method, must provide either regression or classification data')

        print("\nStarting Support Vector Machine")
        if model is SVM.Regular:
            param_grid = handle_auto(model_params, [['C', 1.0],
                                                    ['epsilon', 0.1],
                                                    ['kernel', 'rbf'],
                                                    ['degree', 3],
                                                    ['gamma', 0.0],
                                                    ['coef0', 0.0]])

            if self.__regression:
                self.__svm_reg = grid_search.GridSearchCV(model[1](), param_grid.copy())

            if self.__classification:
                del param_grid['epsilon']
                self.__svm_class = grid_search.GridSearchCV(model[0](), param_grid.copy())

        if model is SVM.Linear:
            param_grid = handle_auto(model_params, [['C', 1.0],
                                                    ['epsilon', 0.1]])
            if self.__regression:
                self.__svm_reg = grid_search.GridSearchCV(model[1](), param_grid.copy())

            if self.__classification:
                del param_grid['epsilon']
                self.__svm_class = grid_search.GridSearchCV(model[0](), param_grid.copy())

        if model is SVM.Nu:
            param_grid = handle_auto(model_params, [['C', 1.0],
                                                    ['nu', 0.5],
                                                    ['kernel', 'rbf'],
                                                    ['degree', 3],
                                                    ['gamma', 0.0],
                                                    ['coef0', 0.0]])

            if self.__regression:
                self.__svm_reg = grid_search.GridSearchCV(model[1](), param_grid.copy())

            if self.__classification:
                del param_grid['C']
                self.__svm_class = grid_search.GridSearchCV(model[0](), param_grid.copy())

        if self.__regression:
            t0 = time()
            self.__svm_reg = self.__svm_reg.fit(self.__x_data, self.__y_data).best_estimator_
            t1 = time()

            print('Time to complete SVM Regression was:\t%.2fs' % (t1 - t0))
            if self.__verbose:
                print(self.__svm_reg)

        if self.__classification:
            t0 = time()
            self.__svm_class = self.__svm_class.fit(self.__x_data, self.__y_labels).best_estimator_
            self.__svm_class_weighted = copy(self.__svm_class)
            self.__svm_class_weighted.class_weight = 'auto'
            self.__svm_class_weighted.fit(self.__x_data, self.__y_labels)
            t1 = time()

            self.__svm_labels = self.__svm_class.predict(self.__x_data)
            self.__svm_labels_weighted = self.__svm_class_weighted.predict(self.__x_data)

            print('Time to complete SVM Classification was:\t%.2fs' % (t1 - t0))
            if self.__verbose:
                print(self.__svm_class)
        self.__svm_params = (model, model_params)
        self.__E_svm = True
        return True

    # Support Vector Machine Get Methods
    @init_check('_Brain__E_svm')
    def get_svm_labels(self, weighted=False):
        if weighted:
            return self.__svm_labels_weighted
        return self.__svm_labels

    @init_check('_Brain__E_svm')
    def get_svm_group(self, label, weighted=False):
        if weighted:
            return self.__x_data[self.__svm_labels_weighted == label]
        return self.__x_data[self.__svm_labels == label]

    @regression_method()
    @init_check('_Brain__E_svm')
    def get_svm_group(self, label, weighted=False):
        if weighted:
            return self.__y_data[self.__svm_labels_weighted == label]
        return self.__y_data[self.__svm_labels == label]

    # Support Vector Machine
    @classification_method()
    @init_check('_Brain__E_svm')
    @data_transform()
    def predict_svm_class(self, x, weighted=False):
        if weighted:
            prediction = self.__svm_class_weighted.predict(x)
        else:
            prediction = self.__svm_class.predict(x)

        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction

    @regression_method()
    @init_check('_Brain__E_svm')
    @data_transform()
    def predict_svm_data(self, x):
        prediction = self.__svm_reg.predict(x)
        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction

    """SVM END"""

    """GMM START"""

    def init_gmm(self, model=GMM.REGULAR, **model_params):
        print('\nStart Gaussian Mixture Model')

        n_components = handle_required(model_params, ['n_components'])

        covariance_type, n_iter, n_init, params, init_params = handle_optional(model_params,
                                                                               [['covariance_type', 'diag'],
                                                                                ['n_iter', 100],
                                                                                ['n_init', 1],
                                                                                ['params', 'wmc'],
                                                                                ['init_params', 'wmc']])

        if model is GMM.REGULAR:
            self.__gmm = model(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, n_init=n_init,
                               params=params, init_params=init_params)

        if (model is GMM.VARIATIONAL) or (model is GMM.VARIATIONAL_INFINITE):
            alpha = handle_optional(model_params, [['alpha', 1]])

            self.__gmm = model(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter, n_init=n_init,
                               params=params, init_params=init_params, alpha=alpha)

        t0 = time()
        self.__gmm.fit(self.__x_data)
        t1 = time()

        self.__gmm_params = (model, model_params)
        self.__gmm_labels = self.__gmm.predict(self.__x_data)

        self.__E_gmm = True
        print('Time to complete Gaussian Mixture Model was:\t%.2fs' % (t1 - t0))
        if self.__verbose:
            print(self.__gmm)
        return True

    @init_check('_Brain__E_gmm')
    @data_transform()
    def predict_gmm_class(self, x):
        prediction = self.__gmm.predict(x)

        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction

    @init_check('_Brain__E_gmm')
    @data_transform()
    def predict_gmm_fuzzy(self, x):
        return self.__gmm.predict_proba(x)

    """GMM END"""


    """Naive Bayes START"""

    def init_naive_bayes(self, model=NAIVE_BAYES.REGULAR, **model_params):
        print('\nStart Naive Bayes')

        if model is NAIVE_BAYES.REGULAR:
            self.__naive_bayes = model()
        if model is NAIVE_BAYES.MULTINOMIAL:
            alpha = handle_optional(model_params, [['alpha', 1]])
            self.__naive_bayes = model(alpha=alpha)
        if model is NAIVE_BAYES.BERNOULLI:
            alpha, binarize = handle_optional(model_params, [['alpha', 1],
                                                             ['binarize', 0.0]])
            self.__naive_bayes = model(alpha=alpha, binarize=binarize)

        t0 = time()
        self.__naive_bayes.fit(self.__x_data, self.__y_labels)
        self.__naive_bayes_labels = self.__naive_bayes.predict(self.__x_data)
        t1 = time()

        self.__naive_bayes_params = (model, model_params)

        self.__E_nb = True
        print('Time to complete Naive Bayes was:\t%.2fs' % (t1 - t0))
        if self.__verbose:
            print(self.__naive_bayes)
        return True


    @init_check('_Brain__E_nb')
    @data_transform()
    def predict_naive_bayes_class(self, x):
        prediction = self.__naive_bayes.predict(x)

        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction

    @init_check('_Brain__E_nb')
    @data_transform()
    def predict_naive_bayes_fuzzy(self, x):
        return self.__naive_bayes.predict_proba(x)

    """Naive Bayes END"""

    @data_transform()
    def predict_all_class(self, x, raw=False, extra=False):
        if len(shape(x)) is 2:
            predict = self.predict_all_class
            return array([predict(data, raw, extra) for data in x])

        output = {}

        if self.__E_cluster:
            output['Cluster'] = self.predict_cluster_class(x)

        if self.__E_knn:
            output['Unweighted Nearest Neighbors'] = self.predict_knn_nearest_class(x)
            output['Weighted Nearest Neighbors'] = self.predict_knn_nearest_class(x, True)
            if extra:
                output['Unweighted Radius Neighbors'] = self.predict_knn_radius_class(x)
                output['Weighted Radius Neighbors'] = self.predict_knn_radius_class(x, True)

        if self.__E_svm:
            output['Unweighted SVM'] = self.predict_svm_class(x, False)
            output['Weighted SVM'] = self.predict_svm_class(x, True)

        if self.__E_gmm:
            output['Gaussian Mixture'] = self.predict_gmm_class(x)

        if self.__E_nb:
            output['Naive Bayes'] = self.predict_naive_bayes_class(x)

        output = oDict(sorted(output.items()))

        if raw:
            return get_dict_values(output)
        return output

    @data_transform()
    def predict_all_class_weighted(self, x, highest=False):
        try:
            self.__weights_class
        except AttributeError:
            print('Must initialize weights first by running calculate_weights()')
            return

        if len(shape(x)) is 2:
            predict = self.predict_all_class_weighted
            return array([predict(data, highest) for data in x])

        pred = self.predict_all_class(x, True)
        weights = self.__weights_class
        pred = swapaxes(vstack((pred, weights)), 0, 1)
        pred = pred[pred[:, 1] != 0]
        pred = array([[n, pred[pred[:, 0] == n][:, 1].sum()] for n in unique(pred[:, 0])])

        if highest:
            return pred[pred[:, 1] == pred[:, 1].max()][0][0]

        return pred

    @regression_method()
    @data_transform()
    def predict_all_data(self, x, raw=False, extra=False):
        if len(shape(x)) is 2:
            predict = self.predict_all_data
            return array([predict(data, raw, extra) for data in x])

        output = {}

        if self.__E_cluster:
            output['Cluster'] = self.predict_cluster_data(x)

        if self.__E_knn:
            output['Unweighted Nearest Neighbors'] = self.predict_knn_nearest_data(x)
            output['Weighted Nearest Neighbors'] = self.predict_knn_nearest_data(x, True)
            if extra:
                output['Unweighted Radius Neighbors'] = self.predict_knn_radius_data(x)
                output['Weighted Radius Neighbors'] = self.predict_knn_radius_data(x, True)

        if self.__E_svm:
            output['SVM'] = self.predict_svm_data(x)

        output = oDict(sorted(output.items()))

        if raw:
            return get_dict_values(output)
        return output

    @regression_method()
    @data_transform()
    def predict_all_data_weighted(self, x, average=False, highest=False):
        try:
            self.__weights_data
        except AttributeError:
            print('Must initialize weights first by running calculate_weights()')
            return

        if len(shape(x)) is 2:
            predict = self.predict_all_data_weighted
            return array([predict(data, highest) for data in x])

        pred = self.predict_all_data(x, True)
        weights = self.__weights_data
        pred = swapaxes(vstack((pred, weights)), 0, 1)
        pred = pred[pred[:, 1] != 0]
        pred = array([[n, pred[pred[:, 0] == n][:, 1].sum()] for n in unique(pred[:, 0])])

        if highest:
            return pred[pred[:, 1] == pred[:, 1].max()][0][0]
        if average:
            pred = pred[:, 0] * pred[:, 1]
            return pred.sum()

        return pred

    @classification_method()
    @data_transform()
    def __calculate_weights_class(self, x_test=None, y_test=None, cutoff=None):
        if x_test is None:
            x_test = self.__x_data
        if y_test is None:
            y_test = self.__y_labels

        names = get_dict_key(self.predict_all_class(x_test[0]))
        predictions = self.predict_all_class(x_test, True)
        if len(predictions.shape) is 1:
            predictions = array([predictions])
        predictions = swapaxes(predictions, 0, 1)
        predictions = array([metrics.accuracy_score(y_test, test) for test in predictions])
        if cutoff is not None:
            predictions[predictions < cutoff] = 0
        predictions /= predictions.sum()
        self.__weights_class = predictions

        print_weights = vstack(predictions)
        names = vstack(names)
        pretty_weights = concatenate((names, print_weights), 1)
        print('Classification Weights:\n' + str(pretty_weights))

    @regression_method()
    @data_transform()
    def __calculate_weights_data(self, x_test=None, y_test=None, cutoff=None, k=500, p=3):
        if x_test is None:
            x_test = self.__x_data
        if y_test is None:
            y_test = self.__y_data

        names = get_dict_key(self.predict_all_data(x_test[0]))
        predictions = self.predict_all_data(x_test, True)
        if len(predictions.shape) is 1:
            predictions = array([predictions])
        predictions = swapaxes(predictions, 0, 1)
        predictions -= y_test
        predictions = abs(predictions)
        predictions **= p
        predictions *= k
        predictions += 1
        predictions = 1 / predictions
        predictions = mean(predictions, 1)
        if cutoff is not None:
            predictions[predictions < cutoff] = 0
        predictions /= predictions.sum()
        self.__weights_data = predictions

        print_weights = vstack(predictions)
        names = vstack(names)
        pretty_weights = concatenate((names, print_weights), 1)
        print('Regressions Weights:\n' + str(pretty_weights))

    @data_transform()
    def calculate_weights(self, x_test=None, y_test_class=None, y_test_data=None, cutoff=None, k=500, p=3):
        """Calculates weights of results based on accuracy, returns a value from 0 - 1 for each value
        Input test points and data. If blank, all data will be used.
        Cuttoff: determines at which point will weight be turned to zero, number from 0 - 1
        K: k constant for equation
        p: Power of x """
        if (not self.__classification) and (not self.__regression):
            print('Weighted combination requires either regression or classification data')
            return

        if self.__classification:
            self.__calculate_weights_class(x_test, y_test_class, cutoff)
        if self.__regression:
            self.__calculate_weights_data(x_test, y_test_data, cutoff, k, p)

    def calculate_score_class(self, y_predicted, y_actual=None, score_type=None):
        """Use this score calculator for more specific scores.
        Use the METRIC enum to choose scoreType, or leave as 'all'"""
        if y_actual is None:
            try:
                y_actual = self.__y_labels
            except AttributeError:
                print('Must provide classification data either when calling function or when making model')
                return
        length = min([len(y_actual), len(y_predicted)])
        if score_type is None:
            score = metrics.classification_report(y_actual[:length], y_predicted[:length])
            print(score)
        else:
            score = score_type(y_actual[:length], y_predicted[:length])
            return score

    def calculate_score_data(self, y_predicted, y_actual=None, k=500, p=3, c=1):
        """Logistic Error Function, Use for general purpose
        Returns a value from (0 - c], Higher Values are better"""
        if y_actual is None:
            try:
                y_actual = self.__y_data
            except AttributeError:
                print('Must provide regression data either when calling function or when making model')
                return
        score = array(y_predicted)
        score -= y_actual
        score = abs(score)
        score **= p
        score *= k
        score += 1
        score = c / score
        score = mean(score)
        return score

    def plot_class(self, method=None, axis=(0, 1), axis_names=("Axis1", "Axis2", "Axis3"), pca=False):
        """Use METHOD enum to choose method for coloring data.
        Set axis to a an array of the axis numbers (starting from 0) or to 'random'
            You can set the axis variable with either 2 or 3 axes
        This method can only be used if your data contains at least 2 axes
        setting usePCA to true is only necessary when your data has more axes than your desired graph type"""

        colors = array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = hstack([colors] * 20)

        if method is CLUSTER:
            y_pred = self.__cluster.labels_.astype(int)
        elif method is SVM:
            y_pred = self.__svm_labels.astype(int)
        elif method is GMM:
            y_pred = self.__gmm_labels.astype(int)
        elif method is NAIVE_BAYES:
            y_pred = self.__naive_bayes_labels.astype(int)
        elif method is None:
            try:
                y_pred = self.__y_labels.astype(int)
            except AttributeError:
                y_pred = zeros((self.__x_data.shape[0],), dtype=int)

        dim = len(axis)
        if dim > 3 or dim < 2:
            print('Number of axes must be either 2 or 3')
            return

        axes = axis
        # if axis is 'random':
        #     axes = random.permutation(array(range(len(self.__XData[0]))))[:dim]

        if pca:
            data = decomposition.RandomizedPCA(dim).fit_transform(self.__x_data)
        else:
            data = self.__x_data

        x_axis = data[:, axes[0]]
        y_axis = data[:, axes[1]]
        if dim == 3:
            z_axis = data[:, axes[2]]

        x_min = min(x_axis)
        x_max = max(x_axis)
        y_min = min(y_axis)
        y_max = max(y_axis)
        if dim == 3:
            z_min = min(z_axis)
            z_max = max(z_axis)

        fig = plt.figure()

        if dim == 2:
            plt.scatter(x_axis, y_axis, color=colors[y_pred].tolist(), s=15)

            # if method is METHOD.Cluster:
            # if hasattr(self.__model, 'cluster_centers_'):
            #        centers = self.__model.cluster_centers_
            #        if usePCA:
            #            centers = decomposition.PCA(dim).transform(centers)
            #        center_colors = colors[:len(centers)]
            #        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(x_min - (0.1 * x_min), x_max + (0.1 * x_max))
            plt.ylim(y_min - (0.1 * y_min), y_max + (0.1 * y_max))
            plt.xticks(())
            plt.yticks(())
            plt.xlabel(axis_names[0])
            plt.ylabel(axis_names[1])
            try:
                plt.suptitle(method.__name__)
            except:
                pass
        else:
            ax = Axes3D(fig)
            ax.scatter(x_axis, y_axis, z_axis, c=colors[y_pred].tolist(), cmap=plt.cm.Paired)
            # if method is METHOD.Cluster:
            # if hasattr(self.__model, 'cluster_centers_'):
            #        centers = self.__model.cluster_centers_
            #        if usePCA:
            #            centers = decomposition.PCA(dim).transform(centers)
            #        center_colors = colors[:len(centers)]
            #        ax.scatter(centers[:, axes[0]], centers[:, axes[1]], centers[:, axes[2]], c=center_colors, s=100)
            try:
                ax.set_title(method.__name__)
            except:
                pass
            ax.set_xlabel(axis_names[0])
            ax.w_xaxis.set_ticklabels([])
            ax.set_ylabel(axis_names[1])
            ax.w_yaxis.set_ticklabels([])
            ax.set_zlabel(axis_names[2])
            ax.w_zaxis.set_ticklabels([])
            ax.autoscale_view()
            ax.autoscale()
        plt.show()

    def plot_regression(self, x_data=None, real=None, predicted=None, axis=(0,), axis_names=("Axis1", "Axis2", "Axis3"), line=False):
        """Use METHOD enum to choose method for coloring data.
        Set axis to a an array of the axis numbers (starting from 0) or to 'random'
            You can set the axis variable with either 2 or 3 axes
        This method can only be used if your data contains at least 2 axes
        setting usePCA to true is only necessary when your data has more axes than your desired graph type"""


        dim = len(axis)
        if dim > 2 or dim < 1:
            print('Number of axes must be either 2 or 3')
            return

        axes = axis

        if x_data is not None:
            data = x_data
        else:
            data = self.__x_data

        if real is not None:
            y_data = real
        else:
            y_data = self.__y_data

        x_axis = data[:, axes[0]]

        if dim == 1:
            y_axis = y_data
        if dim == 2:
            y_axis = data[:, axes[1]]
            z_axis = y_data

        x_min = min(x_axis)
        x_max = max(x_axis)
        y_min = min(y_axis)
        y_max = max(y_axis)
        if dim == 3:
            z_min = min(z_axis)
            z_max = max(z_axis)

        fig = plt.figure()

        if dim == 1:
            if not line:
                plt.scatter(x_axis, y_axis, s=15)
                if predicted is not None:
                    plt.scatter(x_axis, predicted, s=15, c='R')
            else:
                plt.scatter(x_axis, y_axis, s=15)
                if predicted is not None:
                    plt.scatter(x_axis, predicted, s=15, c='R')

            plt.xlim(x_min - (0.1 * x_min), x_max + (0.1 * x_max))
            plt.ylim(y_min - (0.1 * y_min), y_max + (0.1 * y_max))
            plt.xticks(())
            plt.yticks(())
            plt.xlabel(axis_names[0])
            plt.ylabel(axis_names[1])

        else:
            ax = Axes3D(fig)
            ax.scatter(x_axis, y_axis, z_axis, cmap=plt.cm.Paired)
            if predicted is not None:
                ax.scatter(x_axis, y_axis, predicted, cmap=plt.cm.Paired, c='R')


            ax.set_xlabel(axis_names[0])
            ax.w_xaxis.set_ticklabels([])
            ax.set_ylabel(axis_names[1])
            ax.w_yaxis.set_ticklabels([])
            ax.set_zlabel(axis_names[2])
            ax.w_zaxis.set_ticklabels([])
            ax.autoscale_view()
            ax.autoscale()
        plt.show()

    def add_classification_data(self, labels):
        if self.__classification:
            print('You already have classification data, no need to add more!')
            return
        if len(labels) != self.__n_samples:
            print('Length of labels and current X Data do not match.')
            return

        self.__y_labels = labels
        self.__classification = True

    def add_regression_data(self, data):
        if self.__regression:
            print('You already have regression data, no need to add more!')
            return
        if len(data) != self.__n_samples:
            print('Length of data and current X Data do not match.')
            return

        self.__y_labels = data
        self.__regression = True

    @data_transform()
    def update_data(self, new_x, new_labels=None, new_data=None, refit=False):
        # Start: Error Handling
        try:
            if len(new_x[0]) != self.__n_features:
                print('Error! Wrong data size. Length:\t%i Expected:\t%i' % (len(new_x[0]), self.__n_features))
                return
        except:
            print('Input must be in a two dimensional array form')
            return
        # if not check_data(new_x):
        #     return
        # End: Error Handling

        if self.__regression:
            if new_data is not None:
                self.__y_data = concatenate((self.__y_data, new_data), 0)
            else:
                print('Must update regression data along with x data.')
                return

        if self.__classification:
            if new_labels is not None:
                self.__y_labels = concatenate((self.__y_labels, new_labels), 0)
            elif self.__E_est:
                self.__classification = False
            else:
                print('Must update classification data along with x data.')
                return

        self.__x_data = concatenate((self.__x_data, new_x), 0)

        if refit:
            self.refit_models()

    def refit_models(self):
        print('Refitting models')
        if self.__E_est:
            self.init_estimate_labels(self.__est_params[0], **self.__est_params[1])
        if self.__E_cluster:
            self.init_clustering(self.__cluster_params[0], **self.__cluster_params[1])
        if self.__E_knn:
            self.init_neighbors(*self.__knn_params)
        if self.__E_svm:
            self.init_svm(self.__svm_params[0], **self.__svm_params[1])
        if self.__E_gmm:
            self.init_gmm(self.__gmm_params[0], **self.__gmm_params[1])
        if self.__E_nb:
            self.init_naive_bayes(self.__naive_bayes_params[0], **self.__naive_bayes_params[1])
        print('Done Refitting')

    def save_brain(self, file_path='', compress=True):
        externals.joblib.dump(self, file_path, compress)
        print('Successfully saved your brain in: %s' % file_path)

    @staticmethod
    def load_brain(file_path):
        framework = externals.joblib.load(file_path)
        print('Successfully loaded brain from: %s' % file_path)
        return framework

# # import warnings
# #
# # warnings.filterwarnings('error')
# #

def main():
    xData, yData = datasets.make_blobs(1000, 6, 5, random_state=0)
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xData, yData, test_size=0.33, random_state=0)
    brain = Brain(xTrain, y_labels=yTrain, verbose=True)
    brain.init_data_transformation(TRANSFORMATION.Standardize(), TRANSFORMATION.PCA.RandomizedPCA())
    brain.init_clustering(model=CLUSTER.MiniBatch, n_clusters=5)
    brain.init_svm(model=SVM.Nu, nu=arange(0.1, 0.9, 0.1))
    brain.init_gmm(n_components=5)
    brain.init_neighbors()
    brain.init_naive_bayes()
    brain.predict_all_class(xTest[0])
    brain.calculate_weights(xTest, yTest, cutoff=0.5)

if __name__ == "__main__":
    main()