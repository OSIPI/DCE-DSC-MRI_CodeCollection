import pytest
import csv
import pathlib

def osipi_parametrize(arg_names, test_data, xf_labels=None):
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
        The default is None.

    Returns
    -------
    p : pytest.mark,parametrize
    Decorator for parametrizing test function.

    """
    if xf_labels is None:
        xf_labels = []
        
    data = [ case if case[0] not in xf_labels
            else pytest.param(*case, marks=pytest.mark.xfail)
            for case in test_data ]
    
    p = pytest.mark.parametrize(arg_names, data)
    
    return p

def log_init(filename_prefix, filename_label, headers):
    """
    Initialize log file to save reference and measured values from tests
    log file will be saved in test/results

    Parameters
    ----------
    filename_prefix: str
        prefix for the file
    filename_label: str
        label specific for the test
    headers: str
        list of str for the headers of all files

    Returns
    -------

    """
    pathlib.Path('./test/results/').mkdir(parents=True, exist_ok=True)
    filename = './test/results/' + filename_prefix + filename_label + '.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def log_results(filename_prefix, filename_label, row_data):
    """
    write the results from test instance to a log file that was initialized by log_init

    Parameters
    ----------
    filename_prefix: str
        prefix for the file
    filename_label: str
        label specific for the test
    row_data: list
        data that needs to be saved, expects a single row

    Returns
    -------

    """
    filename = './test/results/' + filename_prefix + filename_label + '.csv'
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)
