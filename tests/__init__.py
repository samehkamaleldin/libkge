"""
Although the pytest documentation claims __init__.py files are not necessary in test directories these files are
useful for helping pytest find and correctly import code to be tested. In addition __init__.py in a test directory
is a good place to specify a test-specific logging configuration.
"""
import logging

logging.basicConfig(level=logging.DEBUG)