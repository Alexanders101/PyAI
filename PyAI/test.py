__author__ = 'alex'

import warnings
warnings.filterwarnings("ignore")

from sys import version_info

if version_info[0] == 2: # Python 2.x
    from PyAI import *
    from PyAI import __version__
elif version_info[0] == 3: # Python 3.x
    from PyAI import *
    from PyAI import __version__

import unittest
import sys
import os

np.random.seed(0)
xData, yData = datasets.make_blobs(1000, 6, 5, random_state=0)
xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xData, yData, test_size=0.33, random_state=0)

class ClusterTestCase(unittest.TestCase):

    def test_minibatch_req(self):
        brain = Brain(xTest)
        with self.assertRaises(missing_param_error):
            brain.init_clustering(model=CLUSTER.MiniBatch)

    def test_minibatch_get(self):
        brain = Brain(xTest)
        brain.init_clustering(model=CLUSTER.MiniBatch, n_clusters=5)

        self.assertEqual(type(brain._Brain__cluster), cluster.k_means_.MiniBatchKMeans)

        self.assertTrue((array([2,  9, -7, -8,  -9, 5]) == brain.get_cluster(0)[0].astype(int)).all())

        with self.assertRaises(AttributeError):
            brain.get_cluster_data(0)

        self.assertTrue((brain.get_cluster_labels()[:10] ==[4,2,3,1,3,0,1,1,0,0]).all())

    def test_minibatch_predict(self):
        brain = Brain(xTrain)
        brain.init_clustering(model=CLUSTER.MiniBatch, n_clusters=5)

        self.assertEqual(brain.predict_cluster_class(xTest[0]), 4)
        self.assertTrue((brain.predict_cluster_class(xTest[2:12]) == [1,0,1,3,0,0,3,3,2,1]).all())

        pred = brain.predict_cluster_fuzzy(xTest[0])

        self.assertEqual(pred.shape, (5,2))
        self.assertEqual(pred[:, 1].sum(), 1.0)

        with self.assertRaises(AttributeError):
            brain.predict_cluster_data(xTest[0])

    def test_dbscan(self):
        brain = Brain(xTrain)
        brain.init_clustering(model=CLUSTER.DBScan, eps=5, min_samples=10)

        #Basic Checks
        self.assertFalse(brain.predict_cluster_class(xTest[0]))
        self.assertEqual(len(brain.get_cluster(0)), 124)

        # Outlier Check
        brain.init_clustering(model=CLUSTER.DBScan)
        self.assertTrue((brain.get_cluster_labels() == -1).all())


class SVMTestCase(unittest.TestCase):

    def test_init(self):
        brain = Brain(xTrain)
        with self.assertRaises(AttributeError):
            brain.init_svm()

        brain.add_classification_data(yTrain)

        self.assertTrue(brain.init_svm())

    def test_get(self):
        brain = Brain(xTrain, yTrain)
        self.assertTrue(brain.init_svm(C=[1.0, 2.0, 3.0]))


        self.assertTrue((brain.get_svm_labels()[:10] == [0,1,1,1,4,1,0,1,0,2]).any())

class KNNTestCase(unittest.TestCase):

    def test_init(self):
        brain = brain = Brain(xTrain)

        self.assertTrue(brain.init_neighbors())



def test():
    sys.stdout = open(os.devnull, "w")
    unittest.main()
    sys.stdout = sys.__stdout__

if __name__ == '__main__':
    test()





