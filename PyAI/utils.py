__author__ = 'Alex'
import numpy as np
from math import e
from decorator import decorator


def check_array(input):
    """ Checks to make sure that all members of an array are of the same length.
        Return: List containin the check at [0] and the problem points at [1]
                      True if they are all the same
                      False if they are not, and the problem points in a list"""
    curLen = len(input[0])
    badPoints = np.where(np.array([len(point) for point in input]) != curLen)[0]
    if len(badPoints) is not 0:
        return [False, badPoints]
    return [True, None]


def check_data(inX):
    check, badPoints = check_array(inX)
    if not check:
        print('Error! Data is not consistant: ' + badPoints)
        return False
    return True


def weights_from_distances(distances, power=1, natural=False):
    weights = np.array(distances)
    if natural:
        weights = 1.0 / (e ** (power * weights))
    else:
        weights = 1.0 / (weights ** power)
    perfectPoints = np.where(weights == np.inf)[0]
    weights[weights == np.inf] = 0
    weights /= weights.sum()
    weights[perfectPoints] = 1
    return weights


def weighted_average(X):
    if type(X) is not np.ndarray:
        X = np.array(X)
    return np.sum(X[0, :] * X[1, :])


def weighted_mode(pred, weights, axis=0):
    if axis is None:
        pred = np.ravel(pred)
        weights = np.ravel(weights)
        axis = 0
    else:
        pred = np.asarray(pred)
        weights = np.asarray(weights)
        axis = axis

    if pred.shape != weights.shape:
        weights = np.zeros(pred.shape, dtype=weights.dtype) + weights

    scores = np.unique(np.ravel(pred))  # get ALL unique values
    testshape = list(pred.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)
    for score in scores:
        template = np.zeros(pred.shape)
        ind = (pred == score)
        template[ind] = weights[ind]
        counts = np.expand_dims(np.sum(template, axis), axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent
    return mostfrequent, oldcounts


def get_dict_key(dict, index='all'):
    items = []

    for i in dict.keys():
        items.append(i)

    if index == 'all':
        return items
    else:
        return items[index]


def get_dict_item(dict, index='all'):
    items = []

    for i in dict.items():
        items.append(i[1])

    if index == 'all':
        return items
    else:
        return items[index]


def convert_variable(method):
    return {
        '_Brain__clusterEnabled': 'initClustering()',
        '_Brain__knnEnabled': 'initNeighbors()',
        '_Brain__svmEnabled': 'initSVM()',
        '_Brain__gaussianEnabled': 'initGuassianMixture()',
        '_Brain__neuralnetEnabled': 'initNeuralNet()',
        '_Brain__naiveEnabled': 'initNaiveBayes()',
        '_Brain__semisupEnabled': 'initSemiSupervised()',
    }[method]


def init_check(method):
    @decorator
    def checker(f, self, *args, **kwargs):
        if getattr(self, method, False):
            return f(self, *args, **kwargs)
        else:
            raise AttributeError(
                'ERROR METHOD NOT INITIALIZED\nTrying to run: ' + str(f.__name__) + '()\nPlease run: ' + convert_variable(
                    method) + ' first.')

    return checker


def data_transform():
    @decorator
    def checker(f, self, X, *args, **kwargs):
        newArg = X
        if getattr(self, '_Brain__dataTransformationEnabled', False):
            newArg = getattr(self, '_Brain__manipulator').transform(X)
        return f(self, newArg, *args, **kwargs)

    return checker












