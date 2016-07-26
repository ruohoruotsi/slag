""" File manipulation utility functions """

import cPickle as pickle
import json
import os

def filebase(filepath):
    """For a full filepath like '/path/to/some/file.xyz', return 'file'.

    Like `os.path.basename`, except it also strips the file extension.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    fbase : str

    *** COPIED FROM MARLIB *** 
    """
    return os.path.splitext(os.path.basename(filepath))[0]

def file_check(filename):
    """ check to see if a file exists and is non-empty,
    return error message(s) if not """

    error_msg = None

    if not os.path.exists(filename):
        error_msg = 'ERROR:' + filename + ' does not exist!'

    if not os.path.exists(filename):
        error_msg += '\nERROR:' + filename + ' does not exist!'

    return error_msg

def write_json_file(fname,data):
    """ write data (which should be a dictionary or list)
    to a json file """

    with open(fname,'wb') as outfile:
        json.dump(data,outfile)

def read_json_file(fname,as_strings=True):
    """ read data (which should be a dictionary or list)
    from a json file """

    with open(fname, 'rb') as f_handle:
        data = json.load(f_handle)

    return data

def read_pickle_file(fname):
    """
    Read data from a pickle file.
    """

    data = None
    try:
        f = open(fname,'rb')
        data = pickle.load(f)
        f.close()
    except IOError,details:
        print 'ERROR[utils.read_pickle_file()]: unable to read file',fname
        print details
    # except Exception, details:
    #     import sys
    #     print 'ERROR[utils.read_pickle_file] unexpected error reading file ',\
    #     fname,':',str(details)
    #     print sys.exc_info()[0]
    finally:
        f.close()

    return data

def write_pickle_file(data,fname):
    """
    Save data in a pickle file.
    """

    try:
        f = open(fname,'wb')
        pickle.dump(data,f)
    except IOError,details:
        print 'ERROR[utils.write_pickle_file()]: unable to write to file',fname
        print details
    # except Exception, details:
    #     print 'ERROR[utils.write_pickle_file] unexpected error writing to file ',\
    #       fname,':',str(details)
    #     print sys.exc_info()[0]
    finally:
        f.close()

def read_text_file(fname):
    """
    Read a text file.

    args:
        - fname (string): name of file to read (full path)

    returns:
        - datalines (list of strings): the data from file
    """

    datalines = None
    try:
        fp = open(fname, 'r')
        datalines = fp.readlines()
        fp.close()
    except IOError, details:
        print 'ERROR[utils.read_text_file()]: unable to read file:',fname
        print details
    # except Exception as e:
    #     print 'ERROR[utils.read_text_file()]: unexpected error:', \
    #         sys.exc_info()[0]
    #     print 'Unable to read file:', fname

    return datalines

def write_text_file(data, fname, add_line_break=True):
    """
    Write lines of data to a data file

    args:
        - data (list of strings): text data to be written
        - fname (string): file name
        - add_line_break (bool): add a line break after each line of text
    """

    line = ''
    try:
        f = open(fname, 'w')
        for line in data:
            if add_line_break:
                f.write(line + '\n')
            else:
                f.write(line)
    except IOError, details:
        print 'ERROR[utils.write_text_file()]: unable to write line:', \
            line, ' to file', fname
        print details
    # except Exception, details:
    #     print 'ERROR[utils.write_text_file] unexpected error writing line:', \
    #         line, ' to file ', fname, ':', str(details)
    #     print sys.exc_info()[0]
    finally:
        f.close()
