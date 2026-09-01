# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:08:05 2025

@author: hando
"""
import numpy as np

def convert_value(value):
    """Converts a string into its most suitable Python data type. This is
    useful when parsing data from sources like XML attributes, where all values
    are initially read as strings, but you want to handle them as their correct
    types (e.g., numbers, booleans, or None).

    Parameters
    ----------
    value : str
        The input string to be converted.

    Returns
    -------
    The function returns a single value, and its type is determined by the 
    content of the input string, following a specific order of checks:

    NoneType
        The function returns None if the input string is 'none' 
        (case-insensitive).
    bool
        It returns a boolean value (True or False) if the input string is
        'true' or 'false' (case-insensitive).
    int
        the string consists purely of digits, the function returns an integer.
    float
        If the string contains a decimal point and can be converted to a 
        number, the function returns a float. This is attempted after the
        integer check.
    str
        If none of the above conditions are met, the function returns the
        original string unchanged.
    """

    if value.lower() == 'none':
        return None
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.isdigit():
        return int(value)
    try:
        return float(value)  # Try converting to float
    except ValueError:
        return value  # Return as string if conversion fails
    
def convert_attrib(attrib_dict):
    """ Takes a dictionary of attributes, where the values are strings, and
    returns a new dictionary with the values converted to more appropriate data
    types. It is designed to be used in conjunction with the convert_value
    function.

    Parameters
    ----------
    attrib_dict : dict
        Dictionary-like object, such as the attrib dictionary from an XML
        element, where the keys are strings and the values are also strings.

    Returns
    -------
    dict
        A new dictionary is returned. The keys of this dictionary are the same
        as the input attrib_dict, but each value has been converted into the
        most suitable type (e.g., int, float, bool, or None).
    """

    return {key: convert_value(value) for key, value in attrib_dict.items()}

def convert_time(times, timeunit, mode="multiply"):
    """ Convertis time values between different units and a base unit of
    seconds.

    Parameters
    ----------
    times : list
        List or array of numerical values representing time.
    timeunit : str
        A string specifying the unit of the times input. Supported units are:
            "y" or "a" for years
            "m" for months
            "d" for days
            "h" for hours
    mode : str, optional
        optional string parameter controls the direction of the conversion.
        The default is "multiply".
            "multiply" converts the input times from the specified timeunit
            to seconds.
            "divide" converts the input times from seconds to the specified
            timeunit.

    Returns
    -------
    np.array
        returns a NumPy array with the converted time values.
    """

    timeunit = timeunit
    times_array = np.array(times)
    if timeunit in ["y", "a", "year"]:   
        factor = 60 * 60 * 24 * 365
    if timeunit in ["m", "month"]:
        factor = 60 * 60 * 24 * 365 / 12
    elif timeunit in ["d", "day"]:
        factor = 60 * 60 * 24
    elif timeunit in ["h", "hour"]:
        factor = 60 * 60
    elif timeunit in ["min", "minute"]:
        factor = 60 
    elif timeunit in ["s", "sec", "second"]:
        factor = 1
    if mode == "multiply":
        return times_array * factor
    if mode == "divide":
        return times_array / factor

def split_text(text, cast=None):
    """Safely splits the text content of an XML element into a list of
    tokens, tolerating arbitrary/irregular whitespace (spaces, tabs,
    newlines, indentation) and missing text.

    This replaces the fragile ``element.text.split(" ")`` pattern used
    throughout the XML parsing code. That pattern assumes tokens are
    separated by exactly one space and that ``.text`` is never ``None``,
    which breaks (silently, or with confusing errors like
    ``UnboundLocalError``) whenever an XML file is pretty-printed/indented,
    uses tabs, or has a self-closing/empty element.

    Parameters
    ----------
    text : str or None
        The raw ``.text`` attribute of an ``xml.etree.ElementTree.Element``.
        ``None`` is handled gracefully (e.g. for self-closing tags, or tags
        with no text content).
    cast : callable, optional
        If provided, each token is passed through this callable (e.g.
        ``float``, ``int``) after splitting. Defaults to None, which returns
        the tokens as plain strings.

    Returns
    -------
    list
        A list of whitespace-separated tokens (optionally cast), or an
        empty list if `text` is None or contains only whitespace.
    """

    if text is None:
        return []

    # str.split() with no arguments splits on *any* run of whitespace
    # (spaces, tabs, newlines) and ignores leading/trailing whitespace,
    # unlike str.split(" ") which produces empty strings for such cases.
    tokens = text.split()

    if cast is not None:
        return [cast(token) for token in tokens]

    return tokens
