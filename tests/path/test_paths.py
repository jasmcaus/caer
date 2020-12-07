#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

import os 
import caer 

def test_osname():
    s = caer.path.osname

    assert s == os.name()


def test_cwd():
    s = caer.path.cwd()

    assert s == os.getcwd()


def test_abspath():
    s = caer.path.abspath()

    assert s == os.path.abspath()
    

def test_dirname():
    s = caer.path.dirname(__file__)

    assert s == os.path.dirname(__file__)