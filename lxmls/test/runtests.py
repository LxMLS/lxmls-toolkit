#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, LXMLS_ROOT)

pytest.main(['test_day1.py'])
pytest.main(['test_day2.py'])
pytest.main(['test_day3.py'])
pytest.main(['test_day4.py'])
pytest.main(['test_day5.py'])
pytest.main(['test_day6.py'])
