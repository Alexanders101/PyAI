__author__ = 'Alex'
from numpy import *
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
    weights = array(distances)
    if 0 in weights:
        weights = weights == 0
        weights = weights.astype(float)
        weights /= weights.sum()
        return weights
    if natural:
        weights = 1.0 / (e ** (power * weights))
    else:
        weights = 1.0 / (weights ** power)
    perfectPoints = where(weights == inf)[0]
    # weights[weights == inf] = 0
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


def get_dict_key(dict):

    items = np.array([key for key in dict.keys()])

    return items


def get_dict_values(dict):

    items = np.array([value for value in dict.values()])

    return items


def convert_variable(method):
    return {
        '_Brain__E_cluster': 'init_clustering',
        '_Brain__E_knn': 'initNeighbors()',
        '_Brain__E_smv': 'initSVM()',
        '_Brain__E_gm': 'initGuassianMixture()',
        '_Brain__E_neural': 'initNeuralNet()',
        '_Brain__E_nb': 'initNaiveBayes()',
        '_Brain__E_ss': 'initSemiSupervised()',
    }[method]


def init_check(method):
    @decorator
    def checker(f, self, *args, **kwargs):
        if getattr(self, method, False):
            return f(self, *args, **kwargs)
        else:
            raise AttributeError(
                '\nMETHOD NOT INITIALIZED\nTrying to run: ' + str(f.__name__) + '()\nPlease run: ' + convert_variable(
                    method) + ' first.')

    return checker


def classification_method():
    @decorator
    def checker(f, self, *args, **kwargs):
        method = '_Brain__classification'
        if getattr(self, method, False):
            return f(self, *args, **kwargs)
        else:
            raise AttributeError(
                '\nREGRESSION DATA NOT PROVIDED'
                '\nTrying to run: ' + str(f.__name__) + '()' +
                '\nPlease provide classification data or run \'init_estimate_labels()\'first.')

    return checker

def regression_method():
    @decorator
    def checker(f, self, *args, **kwargs):
        method = '_Brain__regression'
        if getattr(self, method, False):
            return f(self, *args, **kwargs)
        else:
            raise AttributeError(
                '\nREGRESSION DATA NOT PROVIDED'
                '\nTrying to run: ' + str(f.__name__) + '()' +
                '\nPlease provide regression data first.')

    return checker


# def data_transform():
#     @decorator
#     def checker(f, self, X, *args, **kwargs):
#         newArg = X
#         if getattr(self, '_Brain__data_transformed', False):
#             # print('Transformed')
#             newArg = getattr(self, '_Brain__manipulator').transform(X)
#         # else:
#             # print('Not Transformed')
#         return f(self, newArg, *args, **kwargs)
#
#     return checker

def data_transform():
    @decorator
    def checker(f, self, X, *args, **kwargs):
        newArg = X
        already = getattr(self, '_Brain__cur_data_transformed')
        if (not already) and getattr(self, '_Brain__data_transformed', False):
                # print('Transformed')
                if X is not None:
                    newArg = getattr(self, '_Brain__manipulator').transform(X)
                setattr(self, '_Brain__cur_data_transformed', True)
        # else:
            # print('Not Transformed')
        res = f(self, newArg, *args, **kwargs)
        if not already:
            setattr(self, '_Brain__cur_data_transformed', False)
        return res
    return checker


class missing_param_error(Exception):
    def __init__(self, name):
        self.value = "Missing required parameter:\t'%s'" % (name)
    def __str__(self):
        return self.value

def handle_optional(model_params, options = []):
    params = []
    for key, value in options:
        try:
            curr = model_params[key]
            print("'{}' is set to modified value of: {}".format(key, curr))
        except KeyError:
            curr = value
            # print("'{}' is set to default value of: {}".format(key, value))
        params.append(curr)
    return params if len(params) > 1 else params[0]

def handle_required(model_params, options = []):
    params = []
    for key in options:
        try:
            curr = model_params[key]
        except KeyError:
            raise missing_param_error(key)
        params.append(curr)
    return params if len(params) > 1 else params[0]

def handle_auto(model_params, options = []):
    params = {}
    for key, value in options:
        try:
            curr = model_params[key]
            print("'{}' is set to modified value of: {}".format(key, curr))
        except KeyError:
            curr = value
            # print("'{}' is set to default value of: {}".format(key, value))
        if (type(curr) is not list) and (type(curr) is not tuple) and (type(curr) is not ndarray):
            curr = [curr]
        params[key] = curr
    return params
















