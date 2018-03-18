#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, LXMLS_ROOT)

pytest.main(['test_galton_dataset.py', 'test_linear_classifier.py', 'test_sequence_models.py'])
