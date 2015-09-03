__author__ = 'alex'

from sys import version_info
if version_info[0] == 2: # Python 2.x
    from test import test
    from PyAI import *
elif version_info[0] == 3: # Python 3.x
    from PyAI.test import test
    from PyAI.PyAI import *


