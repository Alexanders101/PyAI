#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Alex'
__version__ = 0.91

import warnings
warnings.filterwarnings("ignore")

from sys import version_info
if version_info[0] == 2: # Python 2.x
    from utils import *
elif version_info[0] == 3: # Python 3.x
    from PyAI.utils import *


from sklearn import *
from scipy import stats
from operator import itemgetter
from collections import OrderedDict as oDict
from enum import Enum
from mpl_toolkits.mplot3d import Axes3D
import time
import copy
import matplotlib.pyplot as plt

class CLUSTER_METHOD(Enum):  # Enum for model types used in outputs from Brain
    Cluster = 0
    Neighbors = 1
    SupportVector = 2
    Gaussian = 3
    NeuralNet = 4
    NaiveBayes = 5
    SemiSupervised = 6
    Actual = 8
    Nothing = 7

    @staticmethod
    def getName(method):
        if method is CLUSTER_METHOD.Cluster:
            return 'Cluster'
        if method is CLUSTER_METHOD.Neighbors:
            return 'Nearest Neighbors'
        if method is CLUSTER_METHOD.SupportVector:
            return 'Support Vector Machine'
        if method is CLUSTER_METHOD.Gaussian:
            return 'Gaussian Mixture'
        if method is CLUSTER_METHOD.NeuralNet:
            return 'Neural Net'
        if method is CLUSTER_METHOD.NaiveBayes:
            return 'Naive Bayes'
        if method is CLUSTER_METHOD.SemiSupervised:
            return 'Semi Supervised'
        if method is CLUSTER_METHOD.Nothing:
            return 'Normal'
class MODEL:
    KMean = cluster.KMeans
    MiniBatch = cluster.MiniBatchKMeans
    AffProp = cluster.AffinityPropagation
    MShift = cluster.MeanShift
    Spectral = cluster.SpectralClustering
    Ward = cluster.Ward
    Agglomerative = cluster.AgglomerativeClustering
    DBScan = cluster.DBSCAN
    class SVM:
        SVC = svm.SVC
        LinearSVC = svm.LinearSVC
        NuSVC = svm.NuSVC
        SVR = svm.SVR
        NuSVR = svm.NuSVR

    @staticmethod
    def iterate():
        models = [MODEL.KMean, MODEL.MiniBatch, MODEL.AffProp, MODEL.MShift, MODEL.Spectral, MODEL.Ward,
                  MODEL.Agglomerative, MODEL.DBScan]
        return iter(models)
class SCORING:
    Precision = metrics.precision_score
    F1Score = metrics.f1_score
    RecallScore = metrics.recall_score
    ZeroOne = metrics.zero_one_loss

    class REGRESSION:
        RootMeanSquare = metrics.mean_squared_error
        MeanAbsolute = metrics.mean_absolute_error
        ExplainedVariance = metrics.explained_variance_score
        RSquared = metrics.r2_score

    class CLUSTER:
        AdjustedRandom = metrics.cluster.adjusted_rand_score
        Completeness = metrics.cluster.completeness_score
        Homogeneity = metrics.cluster.homogeneity_score
        VMeasure = metrics.cluster.v_measure_score
class AUTO:
    Grid = grid_search.GridSearchCV
    Random = grid_search.RandomizedSearchCV
class DATA_MANIPULATION:
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
        return ('variance_threshhold', feature_selection.VarianceThreshold(threshold))

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


class _ClusterHold:
    def __init__(self, model, modelHold, modelType, nClusters, labels, modelAverages):
        self.model = model
        self.modelHold = modelHold
        self.modelType = modelType
        self.nClusters = nClusters
        self.labels = labels
        self.modelAverages = modelAverages

    def getAll(self):
        return [self.model, self.modelHold, self.modelType, self.nClusters, self.labels, self.modelAverages]


class Trainer:
    def __init__(self, xData, labels):
        self.nClusters = len(np.unique(labels))
        self.XVal = np.array(xData)
        self.YVal = np.array(labels)
        self.Centers = self.__findCenters()

    @staticmethod
    def __calculateAverage(points):
        numElem = len(points[0])
        numPoints = len(points)
        averagePoint = [0.0] * numElem
        for i in iter(points):
            for j in range(len(i)):
                averagePoint[j] += i[j]
        for k in range(numElem):
            averagePoint[k] = round(float(averagePoint[k] / numPoints), 2)
        return averagePoint

    def __findCenters(self):
        averages = np.empty((self.nClusters,len(self.XVal[0])))
        yVal = self.YVal
        xVal = self.XVal
        separatedList = [xVal[yVal == x] for x in np.unique(yVal)]
        for j in range(self.nClusters):
            averages[j] = self.__calculateAverage(separatedList[j])
        return averages

    def __repr__(self):
        return self.Centers

    def getCenters(self):
        return self.Centers


class Brain:
    """
    This is the main part of the brain. Input data when initializing the Brain.
    Then run the init methods of whichever methods of prediction you wish to use.
    """
    __predictionClusterTypes = np.array(
        [cluster.KMeans, cluster.MiniBatchKMeans, cluster.AffinityPropagation, cluster.MeanShift])
    __version__ = '0.9'

    def __init__(self, init_x, init_y=None, supervised=False, labels=None, data_manipulation=None, median=False):
        # Start: Error Handling
        if len(init_x) == 0 or len(init_x[0]) == 0:
            print('Error! No Data')
            return
        if not check_data(init_x):
            return
        # End: Error Handling

        self.__averageFunction = np.mean
        if median:
            self.__averageFunction = np.median

        self.__XData = np.array(init_x)
        self.__YData = None
        if init_y is not None:
            self.__YData = np.array(init_y)
        self.__Labels = labels
        if (labels is None) and supervised:
            self.__Labels = self.__YData
        self.__nDataPoints = len(init_x)
        self.__nElements = len(init_x[0])
        self.__weights = None

        self.__dataTransformationEnabled = False
        if data_manipulation is not None:
            self.__init_data_handling(data_manipulation)

        self.__clusterEnabled = False
        self.__knnEnabled = False
        self.__svmEnabled = False
        self.__gaussianEnabled = False
        self.__neuralNetEnabled = False
        self.__naiveEnabled = False
        self.__semiSupervisedEnabled = False
        self.__labelsPredicted = False

    # Method Initializations
    def init_clustering(self, nClusters=8, initCenters='k-means++', model=cluster.KMeans, paramRange=None):
        """Use nClusters for Params or set it to \'auto\' for: AffinityPropagation, MeanShift, and DBSCAN models:
        AffinityPropagation: Completely Automatic
        MeanShift:  Bandwidth = [idk - idk]
                    min_bin_frequency = [1 - Infinite], 1
        DBSCAN:     eps = [idk - idk], 0.5
                    min_samples = [1 - infinity], 5"""
        self.__clusterEnabled = True
        initX = self.__XData
        self.__nClusters = nClusters
        self.__modelHold = model

        if type(model()) is cluster.KMeans:
            self.__modelType = 'KMeans Clustering'
            self.__model = model(nClusters, initCenters)
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()

        elif type(model()) is cluster.MiniBatchKMeans:
            self.__modelType = 'Mini Batch KMeans Clustering'
            self.__model = model(nClusters, initCenters)
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()

        elif type(model()) is cluster.AffinityPropagation:
            self.__modelType = 'Affinity Propagation Clustering'
            self.__model = model()
            if nClusters is 'auto':
                self.__auto_calculate_params(CLUSTER_METHOD.Cluster, paramRange)
            elif type(nClusters) is list:
                self.__model = model(nClusters[0])
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()
            self.__nClusters = np.max(self.__model.labels_) + 1

        elif type(model()) is cluster.MeanShift:
            self.__modelType = 'Mean Shift Clustering'
            self.__model = model()
            if nClusters is 'auto':
                self.__auto_calculate_params(CLUSTER_METHOD.Cluster, paramRange)
            elif type(nClusters) is list:
                self.__model = model(nClusters[0], min_bin_freq=nClusters[1])
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()
            self.__nClusters = np.max(self.__model.labels_) + 1

        elif type(model()) is cluster.SpectralClustering:
            self.__modelType = 'Spectral Clustering'
            self.__model = model(nClusters)
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()
            print('Time to complete ' + self.__modelType + ' was:\t%.2fs' % (t1 - t0))

        elif type(model()) is cluster.Ward:
            self.__modelType = 'Ward Clustering'
            self.__model = model(nClusters)
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()

        elif type(model()) is cluster.AgglomerativeClustering:
            self.__modelType = 'Agglomerative Clustering'
            self.__model = model(nClusters)
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()

        elif type(model()) is cluster.DBSCAN:
            self.__modelType = 'Density Based Clustering'
            self.__model = model()
            if nClusters is 'auto':
                self.__auto_calculate_params(CLUSTER_METHOD.Cluster, paramRange)
            elif type(nClusters) is list:
                self.__model = model(nClusters[0], nClusters[1])
            t0 = time.time()
            self.__model.fit(initX)
            t1 = time.time()
            self.__nClusters = np.max(self.__model.labels_) + 1

        self.__cluster_labels = self.__model.labels_
        self.__modelAverages = None
        if self.__YData is not None:
            self.__calculate_centroid_averages()
        print('Time to complete ' + self.__modelType + ' was:\t%.2fs' % (t1 - t0))
    def init_guess_labels(self, nClusters=8, initCenters='k-means++', model=cluster.KMeans, paramRange=None):
        """Used to create labels from clustering algorithms for supervised learning
        Use nClusters for Params or set it to \'auto\' for: AffinityPropagation, MeanShift, and DBSCAN models:
        AffinityPropagation: Completely Automatic
        MeanShift:  Bandwidth = [idk - idk]
                    min_bin_frequency = [1 - Infinite], 1
        DBSCAN:     eps = [idk - idk], 0.5
                    min_samples = [1 - infinity], 5"""
        self.__labelsPredicted = True
        self.__guessHold = (nClusters, initCenters, model, paramRange)

        # Save Clustering model if it exists
        if self.__clusterEnabled:
            hold = _ClusterHold(self.__model, self.__modelHold, self.__modelType,
                                self.__nClusters, self.__cluster_labels, self.__modelAverages)
        initX = self.__XData
        self.__nClusters = nClusters
        self.__modelHold = model

        if type(model()) is cluster.KMeans:
            self.__modelType = 'KMeans Clustering'
            self.__model = model(nClusters, initCenters)

        elif type(model()) is cluster.MiniBatchKMeans:
            self.__modelType = 'Mini Batch KMeans Clustering'
            self.__model = model(nClusters, initCenters)

        elif type(model()) is cluster.AffinityPropagation:
            self.__modelType = 'Affinity Propagation Clustering'
            self.__model = model()
            if nClusters is 'auto':
                self.__auto_calculate_params(CLUSTER_METHOD.Cluster, paramRange)
            elif type(nClusters) is list:
                self.__model = model(nClusters[0])

        elif type(model()) is cluster.MeanShift:
            self.__modelType = 'Mean Shift Clustering'
            self.__model = model()
            if nClusters is 'auto':
                self.__auto_calculate_params(CLUSTER_METHOD.Cluster, paramRange)
            elif type(nClusters) is list:
                self.__model = model(nClusters[0], min_bin_freq=nClusters[1])

        elif type(model()) is cluster.SpectralClustering:
            self.__modelType = 'Spectral Clustering'
            self.__model = model(nClusters)

        elif type(model()) is cluster.Ward:
            self.__modelType = 'Ward Clustering'
            self.__model = model(nClusters)

        elif type(model()) is cluster.AgglomerativeClustering:
            self.__modelType = 'Agglomerative Clustering'
            self.__model = model(nClusters)

        elif type(model()) is cluster.DBSCAN:
            self.__modelType = 'Density Based Clustering'
            self.__model = model()
            if nClusters is 'auto':
                self.__auto_calculate_params(CLUSTER_METHOD.Cluster, paramRange)
            elif type(nClusters) is list:
                self.__model = model(nClusters[0], nClusters[1])

        t0 = time.time()
        self.__model.fit(initX)
        t1 = time.time()

        self.__Labels = self.__model.labels_
        if self.__YData is None:
            self.__YData = self.__Labels

        # Reset Clustering model and clean up
        if self.__clusterEnabled:
            self.__model, self.__modelHold, self.__modelType, self.__nClusters, \
            self.__cluster_labels, self.__modelAverages = hold.getAll()
        else:
            del self.__model
            del self.__modelHold
            del self.__modelType
            del self.__nClusters
            del self.__cluster_labels
            del self.__modelAverages

        print('Time to complete fake labels was:\t%.2fs' % (t1 - t0))
    def init_neighbors(self, nNeighbors=5, radius=1.0):
        """Unsupervised Method: Nearest Neighbors Initialization Method
            nNeighbors = number of neighbors for n based algorithm
            radius = radius for distance based algorithm"""

        self.__knnEnabled = True

        self.__knn = neighbors.NearestNeighbors(nNeighbors, radius)
        t0 = time.time()
        self.__knn.fit(self.__XData)
        t1 = time.time()
        print('Time to complete Nearest Neighbors was:\t%.2fs' % (t1 - t0))
    def init_SVM(self, model=svm.LinearSVC, options=None, paramRange=10):
        """ Supervised Method: Support Vector Machine
            Options: C : [1-1000], 1.0 OR Nu : (0 - 1], 0.5 (For NuSVC)
                     epsilon : [0 - idk], 0.1 OR Nu : (0 - 1], 0.5 (For SVM OR NuSVM)
                     Kernel : ['rbf', 'linear', 'poly', 'sigmoid'], 'rbf'
                     """
        self.__svmEnabled = True
        t0 = time.time()
        self.__svm = model()
        ydata = self.__Labels
        if (type(model()) is svm.SVR) or (type(model()) is svm.NuSVR):
            ydata = self.__YData
        if options is None:
            print('\tUsing Default Parameter for Support Vector Machine')
        elif options == 'auto':
            self.__auto_calculate_params(CLUSTER_METHOD.SupportVector, paramRange)
        else:
            if type(model()) is svm.LinearSVC:
                self.__svm.C = options[0]
            elif type(model()) is svm.SVC:
                self.__svm.C = options[0]
                self.__svm.kernel = options[1]
            elif type(model()) is svm.NuSVC:
                self.__svm.nu = options[0]
                self.__svm.kernel = options[1]
            elif type(model()) is svm.SVR:
                self.__svm.C = options[0]
                self.__svm.epsilon = options[1]
                self.__svm.kernel = options[2]
            elif type(model()) is svm.NuSVR:
                self.__svm.C = options[0]
                self.__svm.nu = options[1]
                self.__svm.kernel = options[2]
        self.__svmWeighted = copy.copy(self.__svm)
        self.__svmWeighted.class_weight = 'auto'
        self.__svm.fit(self.__XData, ydata)
        self.__svmWeighted.fit(self.__XData, ydata)
        self.__svmLabels = np.array(self.__svm.predict(self.__XData))
        self.__svmWeightedLabels = np.array(self.__svmWeighted.predict(self.__XData))
        t1 = time.time()
        print('Time to complete Support Vector Machine was:\t%.2fs' % (t1 - t0))
    def init_gaussian_mixture(self, model=mixture.GMM, options=None, paramRange=None):
        """Unsupervised Method: Gaussian Mixture
        Options: n_components = nClusters, 1
                     covariance_type = Â[\'spherical\'Â, \'Âtied\'Â, \'Âdiag\'Â, \'Âfull\'Â], \'diag\'
                     min_covar / alpha = [0.0001 - 0.01], 0.001 / [1-1000], 1.0
                     tol = [0.0001 - 1], 0.01
                     n_iter = [1 - 10000], 100 """

        self.__gaussianEnabled = True

        self.__gmm = model()
        self.__gmmType = type(model())
        t0 = time.time()
        if options is None:
            print('\tUsing Default Parameter for Gaussian Mixture')
        elif options == 'auto':
            if type(model()) is mixture.GMM:
                self.__auto_calculate_params(CLUSTER_METHOD.Gaussian, paramRange)
        else:
            if type(model()) is mixture.GMM:
                self.__gmm.n_components = options[0]
                self.__gmm.covariance_type = options[1]
                self.__gmm.min_covar = options[2]
                self.__gmm.thresh = options[3]
                self.__gmm.n_iter = options[4]
        del self.__gmmType
        self.__gmm.fit(self.__XData)
        self.__gmmLabels = np.array(self.__gmm.predict(self.__XData))
        t1 = time.time()
        print('Time to complete Gaussian Mixture was:\t%.2fs' % (t1 - t0))
    def init_neural_net(self, nComponents=256, estimator=linear_model.LogisticRegression, options=None, paramRange=10):
        """Supervised Method: Neural Network
        Options: (Leave range as an integer for random selections)
                learning_rate = [idk - idk], idk
                n_iter = [idk - idk], idk
                C = [1 - 1000], 1.0
        """
        self.__neuralNetEnabled = True
        self.__estimator = estimator()
        self.__network = neural_network.BernoulliRBM(nComponents)
        self.__neuralNet = pipeline.Pipeline(steps=[('network', self.__network), ('estimator', self.__estimator)])
        t0 = time.time()
        if options is None:
            print('\tUsing Default Parameter for Neural Net')
        elif options == 'auto':
            if type(paramRange) is int:
                self.__auto_calculate_params(CLUSTER_METHOD.NeuralNet, paramRange, AUTO.Random)
            else:
                self.__auto_calculate_params(CLUSTER_METHOD.NeuralNet, paramRange)
        else:
            self.__network.learning_rate = options[0]
            self.__network.n_iter = options[1]
            self.__estimator.C = options[2]

        self.__neuralNet.fit(self.__XData, self.__Labels)
        self.__neuralNetLabels = np.array(self.__neuralNet.predict(self.__XData))
        t1 = time.time()
        print('Time to complete Neural Net was:\t%.2fs' % (t1 - t0))
    def init_naive_bayes(self, model=naive_bayes.GaussianNB):
        """Supervised Method: Naive Bayes"""
        self.__naiveEnabled = True

        self.__naive = model()
        t0 = time.time()
        self.__naive.fit(self.__XData, self.__Labels)
        self.__naiveLabels = np.array(self.__naive.predict(self.__XData))
        t1 = time.time()
        print('Time to complete Naive Bayes Model was:\t%.2fs' % (t1 - t0))
    def init_semi_supervised(self, nNeighbors=7, kernel='rbf', gamma=20, model=semi_supervised.LabelPropagation):
        """Supervised Method: Semi-Supervised Classification"""
        self.__semiSupervisedEnabled = True

        self.__semiSupervised = model(kernel, gamma, nNeighbors)
        t0 = time.time()
        self.__semiSupervised.fit(self.__XData, self.__Labels)
        self.__semiSupervisedLabels = np.array(self.__semiSupervised.predict(self.__XData))
        t1 = time.time()
        print('Time to complete Semi Supervised Model was:\t%.2fs' % (t1 - t0))

    def __calculate_centroid_averages(self):
        self.__modelAverages = np.array(range(self.__nClusters), float)
        npAverage = np.average
        clusterData = self.get_cluster_data
        for i in range(self.__nClusters):
            self.__modelAverages[i] = npAverage(clusterData(i))
    def __init_data_handling(self, transformations=()):
        self.__dataTransformationEnabled = True
        if type(transformations) is not tuple:
            transformations = (transformations)

        self.__manipulator = pipeline.Pipeline(transformations)
        self.__manipulator.fit(self.__XData, self.__YData)
        self.__XData = self.__manipulator.transform(self.__XData)
    def __handle_data(self, data):
        return self.__manipulator.transform(data)

    # Universal Get Methods
    def get_point(self, index='all'):
        if index == 'all':
            return self.__XData
        return self.__XData[index]
    def get_data(self, index='all'):
        if index == 'all':
            return self.__YData
        return self.__YData[index]
    def get_label(self, index='all'):
        if index == 'all':
            return self.__Labels
        return self.__Labels[index]
    def get_cluster_type(self):
        return type(self.__model)

    # Clustering Get Methods
    @init_check('_Brain__clusterEnabled')
    def get_cluster(self, label):
        return self.__XData[self.__cluster_labels == label]
    @init_check('_Brain__clusterEnabled')
    def get_cluster_data(self, label):
        return self.__YData[self.__cluster_labels == label]
    @init_check('_Brain__clusterEnabled')
    def get_cluster_labels(self):
        return self.__cluster_labels
    def __get_centroid_average(self, label):
        return self.__modelAverages[label]
    @data_transform()
    def __get_distances(self, X):
        return self.__model.transform(X)[0]

    # Nearest Neighbor Get Methods
    @data_transform()
    def get_neighbors(self, X, nNeighbors=-1):
        if nNeighbors == -1:
            nNeighbors = self.get_n_neighbors()
        indices = self.__knn.kneighbors(X, nNeighbors)[1][0]
        return itemgetter(indices)(self.__XData)
    @data_transform()
    def get_neighbor_data(self, X, nNeighbors=-1, weighted=False):
        if nNeighbors == -1:
            nNeighbors = self.get_n_neighbors()
        curNeighbors = self.__knn.kneighbors(X, nNeighbors)
        indices = curNeighbors[1][0]
        if weighted:
            weights = weights_from_distances(curNeighbors[0][0])
            return np.array([itemgetter(indices)(self.__YData), weights])
        return itemgetter(indices)(self.__YData)
    def get_n_neighbors(self):
        return self.__knn.n_neighbors

    # Support Vector Machine Get Methods
    def get_SVM_group(self, label, weighted=False):
        if weighted:
            return self.__XData[self.__svmWeightedLabels == label]
        return self.__XData[self.__svmLabels == label]
    def get_SVM_data(self, label, weighted=False):
        if weighted:
            return self.__YData[self.__svmWeightedLabels == label]
        return self.__YData[self.__svmLabels == label]

    # Gaussian Mixture Get Methods
    def get_gaussian_group(self, label):
        return self.__XData[self.__gmmLabels == label]
    def get_gaussian_data(self, label):
        return self.__YData[self.__gmmLabels == label]

    # Neural Net get Methods
    def get_neural_net_group(self, label):
        return self.__XData[self.__neuralNetLabels == label]
    def get_neural_net_data(self, label):
        return self.__YData[self.__neuralNetLabels == label]

    # Naive Bayes Get Methods
    def get_naive_group(self, label):
        return self.__XData[self.__naiveLabels == label]
    def get_naive_data(self, label):
        return self.__YData[self.__naiveLabels == label]

    # SemiSupervised Get Methods
    def get_semi_supervised_group(self, label):
        return self.__XData[self.__semiSupervisedLabels == label]
    def get_semi_supervised_data(self, label):
        return self.__YData[self.__semiSupervisedLabels == label]

    # Prediction Methods
    # Clustering
    @data_transform()
    def predict_cluster(self, X):
        if self.get_cluster_type() in Brain.__predictionClusterTypes:
            prediction = self.__model.predict(X)
        else:
            print(self.__modelType + ' is not able to predict Points')
            return None
            # TODO: Have a long hard think about this
            predModel = copy.copy(self.__model)
            xD = copy.copy(self.__XData)
            X = [X]
            xD = np.concatenate((xD, X), 0)
            predModel.fit(xD)
            prediction = predModel.labels_[(0 - len(X)):]
        if len(prediction) == 1:
            return prediction[0]
        return prediction
    def predict_cluster_data(self, X, weighted=False, average=True):
        prediction = self.predict_cluster(X)
        if prediction is None:
            return None
        if type(prediction) is np.ndarray:
            if weighted:
                distance = self.__get_distances
                getWeights = weights_from_distances
                weightedAve = weighted_average
                clusterAverages = self.__modelAverages
                if average:
                    return np.array(
                        [weightedAve([clusterAverages, getWeights(distance(pred), 3, False)]) for pred in X])
                return np.array([[clusterAverages, getWeights(distance(pred), 3, False)] for pred in X])
            getData = self.get_cluster_data
            if average:
                average = self.__averageFunction
                return np.array([average(getData(pred), 0) for pred in prediction])
            return np.array([getData(pred) for pred in prediction])

        if weighted:
            data = np.array([self.__modelAverages, weights_from_distances(self.__get_distances(X), 3, False)])
            if average:
                return weighted_average(data)
            return data
        if average:
            return self.__averageFunction(self.get_cluster_data(prediction), 0)
        return self.get_cluster_data(prediction)

    # Nearest Neighbors
    @data_transform()
    def predict_nearest_neighbors(self, X, weighted=False, nNeighbors=None):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if len(np.shape(X)) == 1:
            X = X.reshape(1, -1)

        distances, indeces = self.__knn.kneighbors(X, nNeighbors)

        classes_ = np.unique(self.__Labels)
        y = self.__Labels
        if y.ndim == 1:
            y = y.reshape((-1, 1))
            classes_ = [classes_]

        n_outputs = len(classes_)
        n_samples = X.shape[0]

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if not weighted:
                mode, _ = stats.mode(y[indeces, k], axis=1)
            else:
                weights = [weights_from_distances(distance) for distance in distances]
                mode, _ = weighted_mode(y[indeces, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if self.__Labels.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]

        return y_pred
    @data_transform()
    def predict_nearest_neighbors_data(self, X, weighted=False, nNeighbors=None):
        distances, indeces = self.__knn.kneighbors(X, nNeighbors)

        y = self.__YData
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if not weighted:
            y_pred = np.mean(y[indeces], axis=1)
        else:
            weights = [weights_from_distances(distance) for distance in distances]
            y_pred = np.array([(np.average(y[ind, :], axis=0,
                                           weights=weights[i]))
                               for (i, ind) in enumerate(indeces)])
        if self.__YData.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]
        return y_pred
    @data_transform()
    def predict_radius_neighbors(self, X, weighted=False, radius=None):
        if type(X) is not np.ndarray:
            X = np.array(X)
        if len(np.shape(X)) == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        distances, indeces = self.__knn.radius_neighbors(X, radius)
        inliers = [i for i, nind in enumerate(indeces) if len(nind) != 0]
        outliers = [i for i, nind in enumerate(indeces) if len(nind) == 0]

        classes_ = np.unique(self.__Labels)
        y = self.__Labels
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

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            pred_labels = np.array([y[ind, k] for ind in indeces],
                                   dtype=object)
            if not weighted:
                mode = np.array([stats.mode(pl)[0]
                                 for pl in pred_labels[inliers]], dtype=np.int)
            else:
                weights = [weights_from_distances(distance) for distance in distances]
                mode = np.array([weighted_mode(pl, w)[0]
                                 for (pl, w)
                                 in zip(pred_labels[inliers], weights)],
                                dtype=np.int)

            mode = mode.ravel()

            y_pred[inliers, k] = classes_k.take(mode)

        if outliers:
            y_pred[outliers, :] = -1

        if self.__Labels.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]

        return y_pred
    @data_transform()
    def predict_radius_neighbors_data(self, X, weighted=False, radius=None):
        distances, indeces = self.__knn.radius_neighbors(X, radius)

        y = self.__YData
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        if not weighted:
            y_pred = np.array([np.mean(y[index, :], axis=0)
                               for index in indeces])
        else:
            weights = [weights_from_distances(distance) for distance in distances]
            y_pred = np.array([(np.average(y[index, :], axis=0,
                                           weights=weights[i]))
                               for (i, index) in enumerate(indeces)])
        if self.__YData.ndim == 1:
            y_pred = y_pred.ravel()
        if len(y_pred) is 1:
            y_pred = y_pred[0]

        return y_pred

    # Support Vector Machine
    @data_transform()
    def predict_SVM(self, X, weighted=False):
        if weighted:
            prediction = self.__svmWeighted.predict(X)
        else:
            prediction = self.__svm.predict(X)
        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction
    def predict_SVM_data(self, X, weighted=False, average=True):
        prediction = self.predict_SVM(X, weighted)
        if type(prediction) is np.ndarray:
            getData = self.get_SVM_data
            if average:
                average = self.__averageFunction
                return np.array([average(getData(pred, weighted), 0) for pred in prediction])
            return np.array([getData(pred, weighted) for pred in prediction])
        if average:
            return self.__averageFunction(self.get_SVM_data(prediction, weighted), 0)
        return self.get_SVM_data(prediction, weighted)

    # Gaussian Mixture
    @data_transform()
    def predict_gaussian(self, X, probability=False):
        if len(np.shape(X)) is 1:
            X = [X]
        if probability:
            return self.__gmm.predict_proba(X)
        prediction = self.__gmm.predict(X)
        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction
    def predict_gaussian_data(self, X, average=True):
        prediction = self.predict_gaussian(X)
        if type(prediction) is np.ndarray:
            getData = self.get_gaussian_data
            if average:
                average = self.__averageFunction
                return np.array([average(getData(pred), 0) for pred in prediction])
            return np.array([getData(pred) for pred in prediction])
        if average:
            return self.__averageFunction(self.get_gaussian_data(prediction), 0)
        return self.get_gaussian_data(prediction)

    # Neural Network
    @data_transform()
    def predict_neural_net(self, X):
        prediction = self.__neuralNet.predict(X)
        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction
    def predict_neural_net_data(self, X, average=True):
        prediction = self.predict_neural_net(X)
        if type(prediction) is np.ndarray:
            getData = self.get_neural_net_data
            if average:
                average = self.__averageFunction
                return np.array([average(getData(pred), 0) for pred in prediction])
            return np.array([getData(pred) for pred in prediction])
        if average:
            return self.__averageFunction(self.get_neural_net_data(prediction), 0)
        return self.get_neural_net_data(prediction)

    # Naive Bayes
    @data_transform()
    def predict_naive(self, X, probability=False):
        if probability:
            return self.__naive.predict_proba(X)
        prediction = self.__naive.predict(X)
        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction
    def predict_naive_data(self, X, average=True):
        prediction = self.predict_naive(X)
        if type(prediction) is np.ndarray:
            getData = self.get_naive_data
            if average:
                average = self.__averageFunction
                return np.array([average(getData(pred), 0) for pred in prediction])
            return np.array([getData(pred) for pred in prediction])
        if average:
            return self.__averageFunction(self.get_naive_data(prediction), 0)
        return self.get_naive_data(prediction)

    # Semi Supervised
    @data_transform()
    def predict_semi_supervised(self, X, probability=False):
        if probability:
            return self.__semiSupervised.predict_proba(X)
        prediction = self.__semiSupervised.predict(X)
        if len(prediction) is 1:
            prediction = prediction[0]
        return prediction
    def predict_semi_supervised_data(self, X, average=True):
        prediction = self.predict_semi_supervised(X)
        if type(prediction) is np.ndarray:
            getData = self.get_semi_supervised_data
            if average:
                average = self.__averageFunction
                return np.array([average(getData(pred), 0) for pred in prediction])
            return np.array([getData(pred) for pred in prediction])
        if average:
            return self.__averageFunction(self.get_semi_supervised_data(prediction), 0)
        return self.get_semi_supervised_data(prediction)

    # Multi
    def predict_all(self, X, dataOnly=False, extra=False, nNeighbors=-1):
        if len(np.shape(X)) is 2:
            predict = self.predict_all
            prediction = np.array([predict(data, dataOnly, extra, nNeighbors) for data in X])
            return prediction

        output = {}

        if self.__clusterEnabled:
            output['Cluster'] = self.predict_cluster(X)
        if self.__knnEnabled:
            output['Unweighted Nearest Neighbors'] = self.predict_nearest_neighbors(X)
            output['Weighted Nearest Neighbors'] = self.predict_nearest_neighbors(X, True)
            if extra:
                output['Unweighted Radius Neighbors'] = self.predict_radius_neighbors(X)
                output['Weighted Radius Neighbors'] = self.predict_radius_neighbors(X, True)
        if self.__svmEnabled:
            output['Unweighted SVM'] = self.predict_SVM(X, False)
            output['Weighted SVM'] = self.predict_SVM(X, True)
        if self.__gaussianEnabled:
            output['Gaussian Mixture'] = self.predict_gaussian(X)
        if self.__neuralNetEnabled:
            output['Neural Network'] = self.predict_neural_net(X)
        if self.__naiveEnabled:
            output['Naive Bayes'] = self.predict_naive(X)
        if self.__semiSupervisedEnabled:
            output['SemiSupervised'] = self.predict_semi_supervised(X)
        output = oDict(sorted(output.items()))
        if dataOnly:
            return get_dict_item(output)
        return output
    def predict_all_data(self, X, dataOnly=False, extra=False, nNeighbors=-1):
        if len(np.shape(X)) is 2:
            predict = self.predict_all_data
            prediction = np.array([predict(data, dataOnly, extra, nNeighbors) for data in X])
            return prediction

        output = {}

        if self.__clusterEnabled:
            output['Unweighted Cluster'] = self.predict_cluster_data(X, False)
            if extra:
                if self.get_cluster_type() is MODEL.KMean:
                    output['Weighted Cluster'] = self.predict_cluster_data(X, True)
                if self.get_cluster_type() is MODEL.MiniBatch:
                    output['Weighted Cluster'] = self.predict_cluster_data(X, True)

        if self.__knnEnabled:
            output['Unweighted Nearest Neighbors'] = self.predict_nearest_neighbors_data(X)
            output['Weighted Nearest Neighbors'] = self.predict_nearest_neighbors_data(X, True)
            if extra:
                output['Unweighted Radius Neighbors'] = self.predict_radius_neighbors_data(X)
                output['Weighted Radius Neighbors'] = self.predict_radius_neighbors_data(X, True)

        if self.__svmEnabled:
            output['Unweighted SVM'] = self.predict_SVM_data(X, False)
            output['Weighted SVM'] = self.predict_SVM_data(X, True)
        if self.__gaussianEnabled:
            output['Gaussian Mixture'] = self.predict_gaussian_data(X)
            if extra:
                output['Gaussian Mixture Probability'] = self.predict_gaussian_data(X, True)
        if self.__neuralNetEnabled:
            output['Neural Network'] = self.predict_neural_net_data(X)
        if self.__naiveEnabled:
            output['Naive Bayes'] = self.predict_naive_data(X)
            if extra:
                output['Naive Bayes Probability'] = self.predict_naive(X, True)
        if self.__semiSupervisedEnabled:
            output['SemiSupervised'] = self.predict_semi_supervised_data(X)
            if extra:
                output['SemiSupervised Probability'] = self.predict_semi_supervised(X, True)
        output = oDict(sorted(output.items()))
        if dataOnly:
            return get_dict_item(output)
        return output
    def predict_all_data_weighted(self, X):
        if self.__weights is None:
            print("Please run \'calculateWeights\' first")
        else:
            if len(np.shape(X)) == 2:
                weights = self.__weights
                combine = self.combine_weighted_data
                predictions = self.predict_all_data(X, True)
                final = [combine(prediction, weights) for prediction in predictions]
                return final
            else:
                prediction = self.predict_all_data(X, True)
                final = self.combine_weighted_data(prediction, self.__weights)
                return final
    @staticmethod
    def combine_weighted_data(all_predictions, weights):
        weights = np.array(weights)
        total = np.sum(weights)
        weights /= total
        return weighted_average(np.array([all_predictions, weights]))

    # Update Methods
    def update_data(self, newX, newY=None, newLabels=None, reFit=False):
        # Start: Error Handling
        try:
            if len(newX[0]) != self.__nElements:
                print('Error! Wrong data size. Length:\t%i Expected:\t%i' % (len(newX[0]), self.__nElements))
                return
        except:
            print('Input must be in a two dimensional array form')
            return
        if not check_data(newX):
            return
        # End: Error Handling

        self.__XData = np.concatenate((self.__XData, newX), 0)
        if (self.__YData is not None) and (newY is not None):
            self.__YData = np.concatenate((self.__YData, newY), 0)
        if (self.__Labels is not None) and (newLabels is not None):
            self.__Labels = np.concatenate((self.__Labels, newLabels), 0)
        self.__nDataPoints += len(newX)
        if reFit:
            self.refit_models()
    def refit_models(self):
        t0 = time.time()
        if self.__labelsPredicted:
            if len(self.__YData) != len(self.__XData):
                self.__YData = None
            self.init_guess_labels(self.__guessHold[0], self.__guessHold[1], self.__guessHold[2], self.__guessHold[3])
        if self.__clusterEnabled:
            self.__model.fit(self.__XData)
            self.__cluster_labels = self.__model.labels_
            self.__modelAverages = None
            if self.__YData is not None:
                self.__calculate_centroid_averages()
        if self.__knnEnabled:
            self.__knn.fit(self.__XData)
        if self.__svmEnabled:
            ydata = self.__Labels
            if (type(self.__svm) is svm.SVR) or (type(self.__svm) is svm.NuSVR):
                ydata = self.__YData
            self.__svm.fit(self.__XData, ydata)
            self.__svmWeighted.fit(self.__XData, ydata)
            self.__svmLabels = np.array(self.__svm.predict(self.__XData))
            self.__svmWeightedLabels = np.array(self.__svmWeighted.predict(self.__XData))
        if self.__naiveEnabled:
            self.__naive.fit(self.__XData, self.__Labels)
            self.__naiveLabels = np.array(self.__naive.predict(self.__XData))
        if self.__semiSupervisedEnabled:
            self.__semiSupervised.fit(self.__XData, self.__Labels)
            self.__semiSupervisedLabels = np.array(self.__semiSupervised.predict(self.__XData))
        if self.__neuralNetEnabled:
            self.__neuralNet.fit(self.__XData, self.__Labels)
            self.__neuralNetLabels = np.array(self.__neuralNet.predict(self.__XData))
        if self.__gaussianEnabled:
            self.__gmm.fit(self.__XData)
            self.__gmmLabels = np.array(self.__gmm.predict(self.__XData))
        t1 = time.time()
        print('Time to complete ReFit was:\t%.2fs' % (t1 - t0))

    # Misc Methods
    @staticmethod
    def calculate_error(yPredicted, yActual, k=500, p=3, c=1):
        """Logistic Error Function, Use for general purpose
        Returns a value from (0 - c], Higher Values are better"""
        score = np.array(yPredicted)
        score -= yActual
        score = np.abs(score)
        score **= p
        score *= k
        score += 1
        score = c / score
        score = np.mean(score)
        return score
    def calculate_score(self, yPredicted, yActual=None, scoreType='all'):
        """Use this score calculator for more specific scores.
        Use the METRIC enum to choose scoreType, or leave as 'all'"""
        if yActual is None:
            yActual = self.__YData
        length1 = len(yActual)
        length2 = len(yPredicted)
        length = np.min([length1, length2])
        if scoreType == 'all':
            score = metrics.classification_report(yActual[:length], yPredicted[:length])
            print(score)
        else:
            # noinspection PyCallingNonCallable
            score = scoreType(yActual[:length], yPredicted[:length])
            return score

    # Don't you dare touch this, This is some black voodoo right here
    def __auto_calculate_params(self, method, ranges, gridType=AUTO.Grid):
        if method is CLUSTER_METHOD.NeuralNet:
            if gridType is AUTO.Grid:
                grid = gridType(self.__neuralNet, {'network__learning_rate': ranges[0], 'network__n_iter': ranges[1],
                                                   'estimator__C': ranges[2]})
            else:
                grid = gridType(self.__neuralNet, ['network__learning_rate', 'network__n_iter', 'estimator__C'], ranges)
            grid.fit(self.__XData, self.__Labels)
            print('\tAutomatic Fitting has been completed for Neural Net.\n\tBest Score: ' + str(grid.best_score_))
            print('\tBest Parameters: ' + str(grid.best_params_))
            self.__neuralNet = grid.best_estimator_

        if method is CLUSTER_METHOD.SupportVector:
            ydata = self.__Labels
            if type(self.__svm) is svm.LinearSVC:
                grid = gridType(self.__svm, {'C': ranges[0], 'loss': ['l1', 'l2']})
            elif type(self.__svm) is svm.SVC:
                grid = gridType(self.__svm, {'C': ranges[0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']})
            elif type(self.__svm) is svm.NuSVC:
                grid = gridType(self.__svm, {'nu': ranges[0], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']})
            elif type(self.__svm) is svm.SVR:
                grid = gridType(self.__svm,
                                {'C': ranges[0], 'epsilon': ranges[1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']})
                ydata = self.__YData
            elif type(self.__svm) is svm.NuSVR:
                grid = gridType(self.__svm,
                                {'C': ranges[0], 'nu': ranges[1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']})
                ydata = self.__YData
            grid.fit(self.__XData, ydata)
            print('\tAutomatic Fitting has been completed for Support Vector Machine.\n\tBest Score: ' + str(
                grid.best_score_))
            print('\tBest Parameters: ' + str(grid.best_params_))
            self.__svm = grid.best_estimator_

        if method is CLUSTER_METHOD.Gaussian:
            def predict_gaussian_data_temp(brain, X, average=True):
                prediction = brain.__gmm.predict(X)
                if type(prediction) is np.ndarray:
                    getData = brain.get_gaussian_data
                    if average:
                        average = brain.__averageFunction
                        return np.array([average(getData(pred), 0) for pred in prediction])
                    return np.array([getData(pred) for pred in prediction])
                if average:
                    return brain.__averageFunction(brain.get_gaussian_data(prediction), 0)
                return brain.get_gaussian_data(prediction)

            xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(self.__XData, self.__YData, test_size=0.25)
            if self.__gmmType is mixture.GMM:
                grid = grid_search.ParameterGrid(
                    {'n_components': ranges[0], 'covariance_type': ranges[1], 'min_covar': ranges[2],
                     'tol': ranges[3], 'n_iter': ranges[4]})
            if self.__gmmType is mixture.DPGMM:
                grid = grid_search.ParameterGrid(
                    {'n_components': ranges[0], 'covariance_type': ranges[1], 'alpha': ranges[2], 'tol': ranges[3],
                     'n_iter': ranges[4]})
            grid = np.array(list(grid))

            GMM = mixture.GMM
            DPGMM = mixture.DPGMM
            predictData = predict_gaussian_data_temp
            scoreCalc = Brain.calculate_error
            gmmType = self.__gmmType

            bestScore = 0
            bestParams = None
            bestModel = None
            for param in grid:
                if gmmType is GMM:
                    self.__gmm = GMM(param['n_components'], param['covariance_type'], tol=param['tol'],
                                     min_covar=param['min_covar'], n_iter=param['n_iter'])
                if gmmType is DPGMM:
                    self.__gmm = DPGMM(param['n_components'], param['covariance_type'], tol=param['tol'],
                                       alpha=param['alpha'], n_iter=param['n_iter'])
                self.__gmm.fit(self.__XData)
                self.__gmmLabels = np.array(self.__gmm.predict(self.__XData))
                yPred = predictData(self, xTest)
                score = scoreCalc(yPred, yTest)
                if score > bestScore:
                    bestScore = score
                    bestParams = param
                    bestModel = self.__gmm
            print('\tAutomatic Fitting has been completed for Gaussian Mixture.\n\tBest Score: ' + str(bestScore))
            print('\tBest Parameters: ' + str(bestParams))
            self.__gmm = bestModel

        if method is CLUSTER_METHOD.Cluster:  # Not Working Yet
            def predict_cluster_data_temp(brain, X, weighted=False, average=True):
                prediction = brain.__model.predict(X)
                if prediction is None:
                    return None
                if type(prediction) is np.ndarray:
                    if weighted:
                        distance = brain.__get_distances
                        getWeights = weights_from_distances
                        weightedAve = weighted_average
                        clusterAverages = brain.__modelAverages
                        if average:
                            return np.array(
                                [weightedAve([clusterAverages, getWeights(distance(pred), 3, False)]) for pred in X])
                        return np.array([[clusterAverages, getWeights(distance(pred), 3, False)] for pred in X])
                    getData = brain.get_cluster_data
                    if average:
                        average = brain.__averageFunction
                        return np.array([average(getData(pred), 0) for pred in prediction])
                    return np.array([getData(pred) for pred in prediction])

                if weighted:
                    data = np.array([brain.__modelAverages, weights_from_distances(brain.__get_distances(X), 3, False)])
                    if average:
                        return weighted_average(data)
                    return data
                if average:
                    return brain.__averageFunction(brain.get_cluster_data(prediction), 0)
                return brain.get_cluster_data(prediction)

            xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(self.__XData, self.__YData, test_size=0.25)
            AffProp = cluster.AffinityPropagation
            MShift = cluster.MeanShift
            DBScan = cluster.DBSCAN
            modelType = self.get_cluster_type()
            predictData = predict_cluster_data_temp
            scoreCalc = Brain.calculate_error

            if modelType is AffProp:
                grid = grid_search.ParameterGrid({'damping': np.arange(0.5, 1, 0.05)})
            if modelType is MShift:
                grid = grid_search.ParameterGrid({'bandwidth': ranges[0], 'min_bin_freq': ranges[1]})
            if modelType is DBScan:
                grid = grid_search.ParameterGrid({'eps': ranges[0], 'min_samples': ranges[1]})
            grid = np.array(list(grid))

            bestScore = 0
            bestParams = None
            bestModel = None
            for param in grid:
                if modelType is AffProp:
                    self.__model = AffProp(param['damping'])
                if modelType is MShift:
                    self.__model = MShift(param['bandwidth'], min_bin_freq=param['min_bin_freq'])
                if modelType is DBScan:
                    self.__model = DBScan(param['eps'], param['min_samples'])
                self.__model.fit(self.__XData)
                self.__cluster_labels = self.__model.labels_
                yPred = predictData(self, xTest)
                score = scoreCalc(yPred, yTest)
                if score > bestScore:
                    bestScore = score
                    bestParams = param
                    bestModel = self.__model
            print(
                '\tAutomatic Fitting has been completed for ' + self.__modelType + '.\n\tBest Score: ' + str(bestScore))
            print('\tBest Parameters: ' + str(bestParams))
            self.__model = bestModel

    def plot_data(self, method=CLUSTER_METHOD.Nothing, axis=(0, 1), axisNames=("Axis1", "Axis2", "Axis3"), usePCA=False):
        """Use METHOD enum to choose method for coloring data.
        Set axis to a an array of the axis numbers (starting from 0) or to 'random'
            You can set the axis varaible with either 2 or 3 axes
        This method can only be used if your data contains at least 2 axes
        setting usePCA to true is only necesaary when your data has more axes than your desired graph type"""

        colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
        colors = np.hstack([colors] * 20)

        if method is CLUSTER_METHOD.Cluster:
            y_pred = self.__model.labels_.astype(np.int)
        elif method is CLUSTER_METHOD.SupportVector:
            y_pred = self.__svmLabels.astype(np.int)
        elif method is CLUSTER_METHOD.NaiveBayes:
            y_pred = self.__naiveLabels.astype(np.int)
        elif method is CLUSTER_METHOD.SemiSupervised:
            y_pred = self.__semiSupervisedLabels.astype(np.int)
        elif method is CLUSTER_METHOD.Gaussian:
            y_pred = self.__gmmLabels.astype(np.int)
        elif method is CLUSTER_METHOD.NeuralNet:
            y_pred = self.__neuralNetLabels.astype(np.int)
        elif method is CLUSTER_METHOD.Actual:
            y_pred = self.__Labels.astype(np.int)
        else:
            y_pred = np.zeros((self.__nDataPoints,), dtype=np.int)

        dim = len(axis)

        axes = axis
        if axis is 'random':
            axes = np.random.permutation(np.array(range(len(self.__XData[0]))))[:dim]

        if usePCA:
            data = decomposition.RandomizedPCA(dim).fit_transform(self.__XData)
        else:
            data = self.__XData

        xAxis = data[:, axes[0]]
        yAxis = data[:, axes[1]]
        if dim == 3:
            zAxis = data[:, axes[2]]

        xMin = np.min(xAxis)
        xMax = np.max(xAxis)
        yMin = np.min(yAxis)
        yMax = np.max(yAxis)
        if dim == 3:
            zMin = np.min(zAxis)
            zMax = np.max(zAxis)

        fig = plt.figure()

        if dim == 2:
            plt.scatter(xAxis, yAxis, color=colors[y_pred].tolist(), s=15)

            # if method is METHOD.Cluster:
            # if hasattr(self.__model, 'cluster_centers_'):
            #        centers = self.__model.cluster_centers_
            #        if usePCA:
            #            centers = decomposition.PCA(dim).transform(centers)
            #        center_colors = colors[:len(centers)]
            #        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(xMin - (0.1 * xMin), xMax + (0.1 * xMax))
            plt.ylim(yMin - (0.1 * yMin), yMax + (0.1 * yMax))
            plt.xticks(())
            plt.yticks(())
            plt.xlabel(axisNames[0])
            plt.ylabel(axisNames[1])
            plt.suptitle(CLUSTER_METHOD.getName(method))
        else:
            ax = Axes3D(fig)
            ax.scatter(xAxis, yAxis, zAxis, c=colors[y_pred].tolist(), cmap=plt.cm.Paired)
            # if method is METHOD.Cluster:
            # if hasattr(self.__model, 'cluster_centers_'):
            #        centers = self.__model.cluster_centers_
            #        if usePCA:
            #            centers = decomposition.PCA(dim).transform(centers)
            #        center_colors = colors[:len(centers)]
            #        ax.scatter(centers[:, axes[0]], centers[:, axes[1]], centers[:, axes[2]], c=center_colors, s=100)
            ax.set_title(CLUSTER_METHOD.getName(method))
            ax.set_xlabel(axisNames[0])
            ax.w_xaxis.set_ticklabels([])
            ax.set_ylabel(axisNames[1])
            ax.w_yaxis.set_ticklabels([])
            ax.set_zlabel(axisNames[2])
            ax.w_zaxis.set_ticklabels([])
            ax.autoscale_view()
            ax.autoscale()
        plt.show()
    def autoCalculateWeights(self, xTest=None, yActual=None, cutoff=None, k=500, p=3):
        """Calculates weights of results based on accuracy, returns a value from 0 - 1 for each value
        Input test points and data. If blank, all data will be used.
        Cuttoff: determines at which point will weight be turned to zero, number from 0 - 1
        K: k constant for equation
        p: Power of x"""
        if xTest is None:
            xTest = self.__XData
        if yActual is None:
            yActual = self.__YData
        names = get_dict_key(self.predict_all_data(xTest[0]))
        predictions = [get_dict_item(self.predict_all_data(point)) for point in xTest]
        predictions = np.array(predictions)
        predictions = np.swapaxes(predictions, 0, 1)
        predictions -= yActual
        predictions = np.abs(predictions)
        predictions **= p
        predictions *= k
        predictions += 1
        predictions = 1 / predictions
        predictions = self.__averageFunction(predictions, 1)
        if cutoff is not None:
            predictions[predictions <= cutoff] = 0
        self.__weights = predictions
        printWeights = np.vstack(predictions)
        names = np.vstack(names)
        prettyWeights = np.concatenate((names, printWeights), 1)
        print('Weights:\n' + str(prettyWeights))

    @staticmethod
    def speed_test(point, method, times=10000, reps=3):
        """Times a a function from the api with a point."""
        scores = []
        for t in range(reps):
            t0 = time.time()
            for i in range(times):
                method(point)
            t1 = time.time()
            scores.append(t1 - t0)
        averageTime = np.mean(scores)
        timePer = averageTime / times
        units = 'Seconds'
        if timePer < 0.5:
            timePer *= 1000
            units = 'MilliSeconds'
        print('Average: %f Seconds\nTests: %i\nOperations per Test: %i\nAverage: %f %s per operation.' % (
            averageTime, reps, times, timePer, units))
    # Persistence
    def save_brain(self, filePath='', compress=False):
        externals.joblib.dump(self, filePath, compress)
        print('Successfully saved framework in: ' + filePath)
    @staticmethod
    def load_brain(filePath):
        framework = externals.joblib.load(filePath)
        print('Successfully loaded framework from: ' + filePath)
        return framework
