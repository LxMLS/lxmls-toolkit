# Load dataset files

import os

def find(filename):
    path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    if os.path.exists(path):
        return path