import unittest
import numpy as np
import sys
import os

current_script_directory = os.path.dirname(os.path.realpath(__file__))
project_root_directory = os.path.abspath(os.path.join(current_script_directory, '..', '..', '..'))
sys.path.append(project_root_directory)
from grimoireml.Functions.DistanceFunctions import ManhattanDistance  # Assuming this is the correct import path

 