""" Module for adfraud specific errors """

class Error(Exception):
    """ base calss for Exceptions"""
    pass

class NotYetFittedModelError(Error):
    """ Exceptions for case where model not fitted yet

    Attributes:
    - message:explanation of the error
    """
    def __init__(self,message):
        self.message = message
