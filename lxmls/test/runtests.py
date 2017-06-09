#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import pytest

LXMLS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, LXMLS_ROOT)

pytest.main(['test_day1.py', 'test_day2.py', 'test_day3.py', 'test_day4.py', 'test_day5.py', 'test_day6.py'])