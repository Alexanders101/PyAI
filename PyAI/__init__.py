__author__ = 'alex'

from sys import version_info
if version_info[0] == 2: # Python 2.x
    from tests import test
    from PyAI import *
elif version_info[0] == 3: # Python 3.x
    from PyAI.tests import test
    from PyAI.PyAI import *


