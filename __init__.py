import sys
import os

moduleDir, _ = os.path.split(os.path.abspath(__file__))
sys.path.insert(0, moduleDir + '/python/')
sys.path.insert(0, moduleDir + '/build/')
from tsne.tsne_wrapper import tsne_run
