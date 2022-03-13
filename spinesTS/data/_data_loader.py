import re
import pandas as pd
import os
from spinesTS.base._const import DataTS


FILE_PATH = os.path.dirname(__file__)


def _get_it_name(built_in_func, name):
    assert name is not None

    def wrap():
        return built_in_func[name]

    return wrap


class DataReader:
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
    def __init__(self, print_file_list=True):
        self.file_list = sorted(os.listdir(os.path.join(FILE_PATH, './built-in-datasets/')))
        if print_file_list:
            print(
                "Existing CSV file list: \n",
                f"\r{'>> ' * 10}\n",
                '\r    '.join([re.split('\.', self.file_list[i])[0].strip() + '\n' if i != 0
                           else '\r    ' + re.split('\.', self.file_list[i])[0].strip() + '\n'
                           for i in range(len(self.file_list))]),
                f"\r{'<< ' * 10}"
            )

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
        return self.file_list


LoadElectricDataSets = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Electric_Production')
LoadMessagesSentDataSets = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Messages_Sent')
LoadMessagesSentHourDataSets = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Messages_Sent_Hour')
LoadWebSales = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Web_Sales')
LoadSupermarketIncoming = _get_it_name(BuiltInSeriesData(print_file_list=False), 'Supermarket_Incoming')
