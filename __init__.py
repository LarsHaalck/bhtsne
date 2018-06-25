import importlib
import sys
import os

moduleDir,_ = os.path.split(os.path.abspath(__file__))
sys.path.insert(0, moduleDir + '/build/')
sys.path.insert(0, moduleDir)
try:
    importlib.import_module('tsne_cpp')
    print('using cpp version of tsne')
except ImportError:
    importlib.import_module('tsne_python')
    print('using fallback python tsne')
