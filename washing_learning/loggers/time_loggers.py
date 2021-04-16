# Standard libraries
from functools import wraps
import time


def chronometer(function):
    """
    This function aims to be a decorator to capture the time a function take to execute.

    For instance,

    @chronometer
    def sum(*args):
        result = 0
        for element in args:
            result += element
        return result

    >>> sum(1, 2, 3, 4, 5)
    Function 'sum' executed in 0.000002 s
    15
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print(
            f"Function '{function.__name__}' was executed in {(end-start):f,} s",
            function.__name__,
            end - start,
        )
        return result

    return wrapped
