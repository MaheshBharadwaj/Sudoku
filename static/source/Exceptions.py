class ModelNotFoundException(Exception):
    """
    Model has not been trained.
    """
    pass


class InsuffientArguments(Exception):
    """
    Insufficient Arguments passed to main function
    """
    pass

class FileNotFoundException(Exception):
    """
    The file provided does not exist
    """
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return 'FileNotFoundException: ' + self.path + ' does not exist!'