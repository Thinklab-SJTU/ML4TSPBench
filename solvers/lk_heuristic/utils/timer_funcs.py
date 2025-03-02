import time
import logging
from functools import wraps

# create the logger for the module
logger = logging.getLogger(__name__)


def timeit(func):
    """
    a decorator to time a function (mostly used for debugging)

    :param func: the function to be timed
    :type func: function
    :return: the decorated func
    :rtype: func
    """
    @wraps(func)
    def new_func(*args, **kwargs):

        # start time
        start_time = time.time()

        # execute the function
        result = func(*args, **kwargs)

        # compute elapsed time
        elapsed_time = time.time() - start_time

        # log the ending result
        logger.info('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))

        # returns the result of the function
        return result

    # returns the wrapper function
    return new_func
