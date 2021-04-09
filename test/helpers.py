import pytest

def osipi_parametrize(arg_names, test_data, xf_labels=[]):
    """
    Generate parametrize decorator with XFail marks.
    
    Adds XFail mark to any test case whose label is contained in xf_labels.

    Parameters
    ----------
    arg_names: string
        Comma-delimited string of parameter names for the test function.
    test_data : list of tuples
        Input formated as input for pytest parametrize decorator.
        Each tuple contains the parameters corresponding to a single test case.
        Test case labels must be stored in the first tuple element.
    xf_labels : list of strings, optional
        Each member should correspond to a test case label that is expected to
        fail. These cases will be marked as such in the parametrize decorator.
        The default is [].

    Returns
    -------
    p : pytest.mark,parametrize
    Decorator for parametrizing test function.

    """
    data = [ case if case[0] not in xf_labels
            else pytest.param(*case, marks=pytest.mark.xfail)
            for case in test_data ]
    
    p = pytest.mark.parametrize(arg_names, data)
    
    return p
    
