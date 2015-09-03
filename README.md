# PyAI
A machine learning / artifial intelligence framework written 
for python designed to be easy to use and felxible. 
It is based mostly on scikit-learn and numpy, but incorperates
many open-source and personal machine learning libraries.

## Requirements

- Python 2.x
- numpy
- scipy
- scikit-learn
- matplotlib
- decorator


## Installation Instruction

1. `python setup.py build`
2. `sudo python setup.py install`
3. In python: `import PyAI`
4. Test by typing: `PyAI.test()`

## Usage

The main object in the library is the Brain class (PyAI.Brain). With it you access all of the features in the framework.

    brain = PyAI.Brain(x_data=data, y_labels=labels, y_data=reg_data)

This brain object has 2 modes of operation: classification and regression.

- If you wish to perform classification (discrete) prediction, use the y_labels attribute
- If you wish to perform regression (continuous) prediction, use the y_data attribute 
- Or you can also provide both

Then, you must initialize one of the algorithms available by performing:

    brain.init_XXX()
    # For example
    brain.init_clustering(n_clusters=5)

Currently, the available algorithms are

- clustering
- neighbors
- svm
- gmm
- naive_bayes

Then you can apply any number of prediction methods in order to predict using the models

    brain.predict_xxx_yyy
    # For example
    brain.predict_cluster_labels(test_data)
    brain.predict_svm_data(test_data)

The xxx must match on of the algorithms that you have initialized

The yyy can either be 'labels' or 'data' for classification and regression respectively  
