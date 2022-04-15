import re
import pandas as pd
import os
from tabulate import tabulate
from spinesTS.base import DataTS

FILE_PATH = os.path.dirname(__file__)


def _get_it_name(built_in_func, name):
    """Wrapper of built-in datasets

    Parameters
    ----------
    built_in_func : BuiltInSeriesData class
    name : dataset's name

    Returns
    -------
    wrapped func
    """
    assert name is not None

    def wrap():
        return built_in_func[name]

    return wrap


class DataReader:
    """Data reader for table-data

    Parameters
    ----------
    fp : file path
    sep : str, file separator
    **pd_read_csv_kwargs : pandas.read_csv function params

    Returns
    -------
    None
    """
    def __init__(self, fp, sep=',', **pd_read_csv_kwargs):
        self._FILEPATH = os.path.join(FILE_PATH, './built-in-datasets/', fp)
        if not os.path.exists(self._FILEPATH):
            self._FILEPATH = fp
        assert os.path.exists(self._FILEPATH), f'No such file or directory: {self._FILEPATH}'
        self._ds = DataTS(pd.read_csv(self._FILEPATH, sep=sep, **pd_read_csv_kwargs))

    @property
    def dataset(self):
        return self._ds.data

    def __len__(self):
        return len(self._ds.data)

    def __getitem__(self, item):
        return self._ds.data.__getitem__(item)

    def __str__(self):
        return self._ds.__str__()

    def __repr__(self):
        return self._ds.__repr__()


class BuiltInSeriesData:
    """Load the built-in data

    Parameters
    ----------
    print_file_list : bool, whether to print the exists file name list

    Returns
    -------
    None
    """
    def __init__(self, print_file_list=True):
        self.file_list = sorted(os.listdir(os.path.join(FILE_PATH, './built-in-datasets/')))
        if print_file_list:
            table = []
            for i in range(len(self.file_list)):
                _ = []
                _.append(re.split('\.', self.file_list[i])[0].strip())
                _.append(', '.join(self[i].dataset.columns.tolist()))
                table.append(_)
            print(tabulate(table, headers=["table's name", "table's columns"], showindex="always",
                    tablefmt="pretty", colalign=("right","left", "left")))

    def __getitem__(self, item):
        if isinstance(item, int):
            return DataReader(os.path.join(FILE_PATH, './built-in-datasets/',
                                           self.file_list[item]))
        elif isinstance(item, str):
            if not item.endswith('.csv'):
                item = item + '.csv'
            return DataReader(os.path.join(FILE_PATH, './built-in-datasets/',
                                           self.file_list[self.file_list.index(item)]))
        else:
            raise KeyError(f"invalid key: {item}")

    @property
    def names(self):
        """Returns the built-in series data names-list."""
        return self.file_list


LoadElectricDataSets = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Electric_Production')
LoadMessagesSentDataSets = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Messages_Sent')
LoadMessagesSentHourDataSets = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Messages_Sent_Hour')
LoadWebSales = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Web_Sales')
LoadSupermarketIncoming = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Supermarket_Incoming')
