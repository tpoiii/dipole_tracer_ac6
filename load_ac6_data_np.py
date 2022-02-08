"""
provides routines to locate and load AeroCube-6 CSV data files
Originally by Mike Shumko
"""
import pathlib
from datetime import datetime, date
import dateutil.parser

import numpy as np
from numpy.lib.recfunctions import append_fields


def load_ac6_file(sc_id, day, data_type, ac6_dir=f'data',
                 empty_file_warn=True):
    """
    Loads the AC6 data file into a dictionary and parses the date-time columns
    into datetime.datetime objects.

    If the AC6 file has a header but is otherwise empty, this function will
    return an empty dictionary. A status message is printed if 
    empty_file_warn==True.

    Parameters
    ----------
    sc_id: str
        AC6 spacecraft id, either A or B, case insensitive.
    day: str, datetime.datetime, or datetime.date
        The day to load the data from. If string, it must have a format that 
        can be parsed by dateutil.parser.parse.
    data_type: str
        The AC6 data type. Can be either '10Hz', 'coords', 'survey', or 'att'.
        Case sensitive.
    ac6_dir: str or pathlib.Path, optional
        The AC6 data directory.
    empty_file_warn: bool, optional
        Toggles the empty file warning. If True, it will print the filename of
        the empty file. 

    Returns
    -------
    ac6_dict: dictionary
        A dictionary containing the AC6 data with the date and time columns
        converted to datetime.datetime objects with the "dateTime" key.
    """
    path = find_ac6_file(sc_id, day, data_type, ac6_dir)
    with open(path, 'r') as f:
        names = next(f).rstrip().split(',')
        dtypes = 5*[int] + (len(names)-5)*[float]
        try:
            data = np.genfromtxt(f, names=names, delimiter=',', dtype=dtypes)
        except ValueError as err:
            if 'could not assign tuple of length 2 to structure with 35 fields.' in str(err):
                if empty_file_warn:
                    print(f'AC6 file is empty: {path.name}.')
                return {}
            raise

    # Save the data to a dict
    ac6_dict = {key:data[key] for key in data.dtype.names}

    # Convert the date and time columns into datetime.datetime() objects.
    # First convert the fraction of a second to integer microseconds.
    microsecond = np.round(
        # Convert to float microsecond
        1_000_000 * (data['second'] - np.floor(data['second'])), 
        # Round to avoid machine precision inaccuracy: "100000." -> "99999."
        1).astype(int)
    data = append_fields(data, 'microsecond', microsecond, np.int)
    # Save to dict
    time_keys = ['year', 'month', 'day', 'hour', 'minute', 
                'second', 'microsecond']
    ac6_dict['dateTime'] = np.array([
                datetime(*np.array(list(row), dtype=np.int)) 
                for row in data[time_keys]
                ])
    return ac6_dict

def find_ac6_file(sc_id, day, data_type, ac6_dir):
    """
    Uses glob to find the ac6 file and return the absolute path to that file.

    Parameters
    ----------
    sc_id: str
        AC6 spacecraft id, either A or B, case insensitive.
    day: str, datetime.datetime, or datetime.date
        The day to load the data from. If string, it must have a format that 
        can be parsed by dateutil.parser.parse.
    data_type: str
        The AC6 data type. Can be either '10Hz', 'coords', 'survey', or 'att'.
        Case sensitive.
    ac6_dir: str or pathlib.Path
        The AC6 data directory.

    Returns
    -------
    ac6_path : pathlib.Path
        A path to the desired AC6 file. 
    """
    if isinstance(day, str):
        day = dateutil.parser.parse(day)

    glob_str =(f'AC6-{sc_id.upper()}_{datetime.strftime(day, "%Y%m%d")}'
                f'_L2_{data_type}*csv')
    paths_list = list(pathlib.Path(ac6_dir).rglob(glob_str))
    assert len(paths_list) == 1, (f'{len(paths_list)} AC6 data paths '
                                f'found\npaths_list={paths_list}')
    return paths_list[0]

if __name__ == "__main__":
    d = load_ac6_file('a', date(2017, 5, 2), '10Hz')
