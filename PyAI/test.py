__author__ = 'alex'
from sys import version_info
if version_info[0] == 2: # Python 2.x
    from PyAI import *
elif version_info[0] == 3: # Python 3.x
    from PyAI.PyAI import *

from sklearn import *


def test_functions(model=Brain, testPoint=[[1, 1]], testLabel=0):
    print('Starting Test. Version %s' % model.__version__)
    t0 = time.time()
    for i in range(1000):
        model.get_data(0)
        model.get_point(0)
        model.get_cluster(testLabel)
        model.get_cluster_data(testLabel)
        model.get_cluster_labels()
        model.get_cluster_type()
        model.get_label()
        model.get_neighbors(testPoint)
        model.get_neighbor_data(testPoint)
        model.get_neighbor_data(testPoint, weighted=True)
        model.get_SVM_data(testLabel)
        model.get_SVM_data(testLabel, True)
        model.get_SVM_group(testLabel)
        model.get_SVM_group(testLabel, True)
        model.get_neural_net_data(testLabel)
        model.get_neural_net_group(testLabel)
        model.get_gaussian_data(testLabel)
        model.get_gaussian_group(testLabel)
        model.get_naive_data(testLabel)
        model.get_naive_group(testLabel)
        model.get_semi_supervised_data(testLabel)
        model.get_semi_supervised_group(testLabel)
        model.predict_cluster(testPoint)
        model.predict_cluster_data(testPoint)
        model.predict_cluster_data(testPoint, True)
        model.predict_nearest_neighbors(testPoint)
        model.predict_nearest_neighbors(testPoint, True)
        model.predict_nearest_neighbors_data(testPoint)
        model.predict_nearest_neighbors_data(testPoint, True)
        model.predict_SVM(testPoint)
        model.predict_SVM(testPoint, True)
        model.predict_SVM_data(testPoint)
        model.predict_SVM_data(testPoint, True)
        model.predict_gaussian(testPoint)
        model.predict_gaussian_data(testPoint)
        model.predict_neural_net(testPoint)
        model.predict_neural_net_data(testPoint)
        model.predict_naive(testPoint)
        model.predict_naive_data(testPoint)
        model.predict_semi_supervised(testPoint)
        model.predict_semi_supervised_data(testPoint)
        model.predict_all(testPoint)
        model.predict_all_data(testPoint)
        model.predict_all_data_weighted(testPoint)
    t1 = time.time()
    print('test completed successfully in %.2fs' % (t1 - t0))
    return True


def test():
    xData, yData = datasets.make_blobs(1000, 6, 5, random_state=0)

    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xData, yData, test_size=0.33, random_state=0)
    # trainSet = Trainer(xTrain,yTrain)
    transforms = manipulations()
    brain = Brain(xTrain, yTrain, True, data_manipulation=transforms)
    brain.init_clustering(5)
    brain.init_neighbors(2, 1.0)
    brain.init_SVM(model=svm.SVC, options='auto', paramRange=[np.arange(1, 10, 1)])
    brain.init_naive_bayes()
    brain.init_semi_supervised()
    brain.init_gaussian_mixture(options='auto', paramRange=[[5], ['tied'], [0.001], np.arange(0.01, 0.1, 0.01), [400]])
    brain.init_neural_net()
    brain.autoCalculateWeights(xTest, yTest, 0.5)
    if test_functions(brain, xTest[0], 0):
        print('Everything is good')


def manipulations():
    manip = []
    # Add various manipulations here
    manip.append(DATA_MANIPULATION.Standardize())

    if len(manip) is not 0:
        return manip
    return None


if __name__ == '__main__':
    test()
