import pandas as pd
from sklearn.base import TransformerMixin

from spinesTS.utils import check_is_fitted
from spinesTS.base import TableFeatureGenerateMixin


class DateExtendFeatures(TableFeatureGenerateMixin, TransformerMixin):
    def __init__(self, date_col, drop_date_col=True, format=None, columns_prefix='timefeatures_'):
        self.date_col = date_col
        self.drop_date_col = drop_date_col
        self.format = format
        if not isinstance(columns_prefix, str):
            columns_prefix = 'timefeatures_'

        self.columns_prefix = columns_prefix

        self.__spinesTS_is_fitted__ = False

    @staticmethod
    def day_of_quarter(s):
        if 1 <= s.month <= 3:
            return s.dayofyear
        elif 4 <= s.month <= 6:
            return s.dayofyear - (
                    pd.to_datetime(str(s.year) + '-03-31') - pd.to_datetime(str(s.year) + '-01-01')).days - 1
        elif 7 <= s.month <= 9:
            return s.dayofyear - (
                    pd.to_datetime(str(s.year) + '-06-30') - pd.to_datetime(str(s.year) + '-01-01')).days - 1
        else:
            return s.dayofyear - (
                    pd.to_datetime(str(s.year) + '-9-30') - pd.to_datetime(str(s.year) + '-01-01')).days - 1

    @staticmethod
    def day2season(s, season, year_lag=0):
        if season == 'spring':
            time_point = pd.to_datetime(str(s.year - year_lag) + '-03-01')
        elif season == 'summer':
            time_point = pd.to_datetime(str(s.year - year_lag) + '-06-01')
        elif season == 'autumn':
            time_point = pd.to_datetime(str(s.year - year_lag) + '-09-01')
        else:
            time_point = pd.to_datetime(str(s.year - year_lag) + '-12-01')

        return (s - time_point).days

    @staticmethod
    def season(s):
        """0: Winter - 1: Spring - 2: Summer - 3: Fall"""
        if s.month in [12, 1, 2]:
            return 0
        elif s.month in [3, 4, 5]:
            return 1
        elif s.month in [6, 7, 8]:
            return 2
        else:
            return 3

    def fit(self, df):
        self.__spinesTS_is_fitted__ = True

        return self

    def transform(self, df):
        """Date features generation"""
        check_is_fitted(self)

        assert isinstance(df, pd.DataFrame)
        x = df[self.date_col].copy().to_frame()
        ds_col = pd.to_datetime(x[self.date_col], format=self.format)

        # Basic timestamp features
        x[self.columns_prefix + 'hour'] = ds_col.dt.hour
        x[self.columns_prefix + 'minute'] = ds_col.dt.minute
        x[self.columns_prefix + 'weekday'] = ds_col.dt.weekday
        x[self.columns_prefix + 'week'] = ds_col.apply(lambda s: s.week)
        x[self.columns_prefix + 'month'] = ds_col.dt.month
        x[self.columns_prefix + 'quarter'] = ds_col.dt.quarter
        x[self.columns_prefix + 'day_of_month'] = ds_col.dt.day
        x[self.columns_prefix + 'day_of_year'] = ds_col.dt.dayofyear
        x[self.columns_prefix + 'day_of_quarter'] = ds_col.apply(lambda s: self.day_of_quarter(s))
        x[self.columns_prefix + 'week_of_month'] = ds_col.apply(lambda d: (d.day - 1) // 7 + 1)
        x[self.columns_prefix + "season"] = ds_col.apply(lambda s: self.season(s))

        # Boolean Flag feature
        x[self.columns_prefix + 'is_weekend'] = ds_col.apply(lambda s: 0 if s.weekday() in [5, 6] else 1)
        x[self.columns_prefix + 'is_start_of_week'] = ds_col.apply(lambda s: 0 if s.weekday() != 0 else 1)
        x[self.columns_prefix + 'is_end_of_week'] = ds_col.apply(lambda s: 0 if s.weekday() != 4 else 1)
        x[self.columns_prefix + 'is_start_of_month'] = ds_col.dt.is_month_start
        x[self.columns_prefix + 'is_middle_of_month'] = ds_col.apply(lambda s: 0 if s.day != 15 else 1)
        x[self.columns_prefix + 'is_end_of_month'] = ds_col.dt.is_month_end
        x[self.columns_prefix + 'is_quarter_start'] = ds_col.dt.is_quarter_start
        x[self.columns_prefix + 'is_year_start'] = ds_col.dt.is_year_start
        x[self.columns_prefix + 'is_year_end'] = ds_col.dt.is_year_end

        # Time difference feature
        x[self.columns_prefix + 'day_to_mid_quarter'] = x[self.columns_prefix + 'day_of_quarter'] - 15
        x[self.columns_prefix + 'day_to_start_quarter'] = x[self.columns_prefix + 'day_of_quarter'] - 1
        x[self.columns_prefix + 'day_to_end_quarter'] = x[self.columns_prefix + 'day_of_quarter'] - 90
        x[self.columns_prefix + 'day_to_monday'] = abs(ds_col.dt.weekday - 1)  # 仅考虑当周周一
        x[self.columns_prefix + 'day_to_friday'] = abs(ds_col.dt.weekday - 5)  # 仅考虑当周周五
        x[self.columns_prefix + 'day_to_middle_week'] = abs(ds_col.dt.weekday - 3)  # 仅考虑当周周三
        x[self.columns_prefix + 'day_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days
        x[self.columns_prefix + 'hour_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days * 24
        x[self.columns_prefix + 'minute_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days * 24 * 60
        x[self.columns_prefix + 'sec_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days * 24 * 3600
        x[self.columns_prefix + 'month_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days / 30
        x[self.columns_prefix + 'week_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days / 7
        x[self.columns_prefix + 'quarter_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days / 90
        x[self.columns_prefix + 'year_to_start_timestamp'] = (ds_col - ds_col.min()).dt.days / 365

        # season Time difference feature
        x[self.columns_prefix + 'day_to_spring'] = ds_col.apply(lambda s: self.day2season(s, 'spring'))
        x[self.columns_prefix + 'day_to_summer'] = ds_col.apply(lambda s: self.day2season(s, 'summer'))
        x[self.columns_prefix + 'day_to_autumn'] = ds_col.apply(lambda s: self.day2season(s, 'autumn'))
        x[self.columns_prefix + 'day_to_winter'] = ds_col.apply(lambda s: self.day2season(s, 'winter'))
        x[self.columns_prefix + 'week_to_spring'] = ds_col.apply(lambda s: self.day2season(s, 'spring')) / 7
        x[self.columns_prefix + 'week_to_summer'] = ds_col.apply(lambda s: self.day2season(s, 'summer')) / 7
        x[self.columns_prefix + 'week_to_autumn'] = ds_col.apply(lambda s: self.day2season(s, 'autumn')) / 7
        x[self.columns_prefix + 'week_to_winter'] = ds_col.apply(lambda s: self.day2season(s, 'winter')) / 7
        x[self.columns_prefix + 'month_to_spring'] = ds_col.apply(lambda s: self.day2season(s, 'spring')) / 30
        x[self.columns_prefix + 'month_to_summer'] = ds_col.apply(lambda s: self.day2season(s, 'summer')) / 30
        x[self.columns_prefix + 'month_to_autumn'] = ds_col.apply(lambda s: self.day2season(s, 'autumn')) / 30
        x[self.columns_prefix + 'month_to_winter'] = ds_col.apply(lambda s: self.day2season(s, 'winter')) / 30
        x[self.columns_prefix + 'quarter_to_spring'] = ds_col.apply(lambda s: self.day2season(s, 'spring')) / 90
        x[self.columns_prefix + 'quarter_to_summer'] = ds_col.apply(lambda s: self.day2season(s, 'summer')) / 90
        x[self.columns_prefix + 'quarter_to_autumn'] = ds_col.apply(lambda s: self.day2season(s, 'autumn')) / 90
        x[self.columns_prefix + 'quarter_to_winter'] = ds_col.apply(lambda s: self.day2season(s, 'winter')) / 90

        x[self.columns_prefix + 'day_to_next_year_spring'] = ds_col.apply(lambda s: self.day2season(s, 'spring', -1))
        x[self.columns_prefix + 'day_to_next_year_summer'] = ds_col.apply(lambda s: self.day2season(s, 'summer', -1))
        x[self.columns_prefix + 'day_to_next_year_autumn'] = ds_col.apply(lambda s: self.day2season(s, 'autumn', -1))
        x[self.columns_prefix + 'day_to_next_year_winter'] = ds_col.apply(lambda s: self.day2season(s, 'winter', -1))
        x[self.columns_prefix + 'week_to_next_year_spring'] = ds_col.apply(
            lambda s: self.day2season(s, 'spring', -1)) / 7
        x[self.columns_prefix + 'week_to_next_year_summer'] = ds_col.apply(
            lambda s: self.day2season(s, 'summer', -1)) / 7
        x[self.columns_prefix + 'week_to_next_year_autumn'] = ds_col.apply(
            lambda s: self.day2season(s, 'autumn', -1)) / 7
        x[self.columns_prefix + 'week_to_next_year_winter'] = ds_col.apply(
            lambda s: self.day2season(s, 'winter', -1)) / 7
        x[self.columns_prefix + 'month_to_next_year_spring'] = ds_col.apply(
            lambda s: self.day2season(s, 'spring', -1)) / 30
        x[self.columns_prefix + 'month_to_next_year_summer'] = ds_col.apply(
            lambda s: self.day2season(s, 'summer', -1)) / 30
        x[self.columns_prefix + 'month_to_next_year_autumn'] = ds_col.apply(
            lambda s: self.day2season(s, 'autumn', -1)) / 30
        x[self.columns_prefix + 'month_to_next_year_winter'] = ds_col.apply(
            lambda s: self.day2season(s, 'winter', -1)) / 30
        x[self.columns_prefix + 'quarter_to_next_year_spring'] = ds_col.apply(
            lambda s: self.day2season(s, 'spring', -1)) / 90
        x[self.columns_prefix + 'quarter_to_next_year_summer'] = ds_col.apply(
            lambda s: self.day2season(s, 'summer', -1)) / 90
        x[self.columns_prefix + 'quarter_to_next_year_autumn'] = ds_col.apply(
            lambda s: self.day2season(s, 'autumn', -1)) / 90
        x[self.columns_prefix + 'quarter_to_next_year_winter'] = ds_col.apply(
            lambda s: self.day2season(s, 'winter', -1)) / 90

        x[self.columns_prefix + 'day_to_last_year_spring'] = ds_col.apply(lambda s: self.day2season(s, 'spring', 1))
        x[self.columns_prefix + 'day_to_last_year_summer'] = ds_col.apply(lambda s: self.day2season(s, 'summer', 1))
        x[self.columns_prefix + 'day_to_last_year_autumn'] = ds_col.apply(lambda s: self.day2season(s, 'autumn', 1))
        x[self.columns_prefix + 'day_to_last_year_winter'] = ds_col.apply(lambda s: self.day2season(s, 'winter', 1))
        x[self.columns_prefix + 'week_to_last_year_spring'] = ds_col.apply(
            lambda s: self.day2season(s, 'spring', 1)) / 7
        x[self.columns_prefix + 'week_to_last_year_summer'] = ds_col.apply(
            lambda s: self.day2season(s, 'summer', 1)) / 7
        x[self.columns_prefix + 'week_to_last_year_autumn'] = ds_col.apply(
            lambda s: self.day2season(s, 'autumn', 1)) / 7
        x[self.columns_prefix + 'week_to_last_year_winter'] = ds_col.apply(
            lambda s: self.day2season(s, 'winter', 1)) / 7
        x[self.columns_prefix + 'month_to_last_year_spring'] = ds_col.apply(
            lambda s: self.day2season(s, 'spring', 1)) / 30
        x[self.columns_prefix + 'month_to_last_year_summer'] = ds_col.apply(
            lambda s: self.day2season(s, 'summer', 1)) / 30
        x[self.columns_prefix + 'month_to_last_year_autumn'] = ds_col.apply(
            lambda s: self.day2season(s, 'autumn', 1)) / 30
        x[self.columns_prefix + 'month_to_last_year_winter'] = ds_col.apply(
            lambda s: self.day2season(s, 'winter', 1)) / 30
        x[self.columns_prefix + 'quarter_to_last_year_spring'] = ds_col.apply(
            lambda s: self.day2season(s, 'spring', 1)) / 90
        x[self.columns_prefix + 'quarter_to_last_year_summer'] = ds_col.apply(
            lambda s: self.day2season(s, 'summer', 1)) / 90
        x[self.columns_prefix + 'quarter_to_last_year_autumn'] = ds_col.apply(
            lambda s: self.day2season(s, 'autumn', 1)) / 90
        x[self.columns_prefix + 'quarter_to_last_year_winter'] = ds_col.apply(
            lambda s: self.day2season(s, 'winter', 1)) / 90

        df = pd.concat((df, x), axis=1)
        return df if not self.drop_date_col else df.drop(columns=self.date_col)
