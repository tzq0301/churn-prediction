from typing import List, Dict, Any

import pandas as pd
from pandas import DataFrame
import structlog

train_data_filepath = 'data/train.csv'
test_data_filepath = 'data/test.csv'


def load_data(filepath: str) -> DataFrame:
    return pd.read_csv(filepath, index_col=0)


def drop_useless_columns(df: DataFrame, columns: List[str]) -> None:
    for column in columns:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)


def drop_nan_rows(df: DataFrame, columns: List[str]) -> None:
    df.dropna(subset=columns, inplace=True)


def make_enum_columns_be_numeric(df: DataFrame, mappings: Dict[str, Dict[Any, int]]) -> None:
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)


if __name__ == '__main__':
    log = structlog.get_logger()

    train_data: DataFrame = load_data(train_data_filepath)
    log.msg('loaded train data', filepath=train_data_filepath, total=len(train_data))
    test_data: DataFrame = load_data(test_data_filepath)
    log.msg('loaded test data', filepath=train_data_filepath, total=len(test_data))

    useless_columns: List[str] = ['Surname', 'CustomerId']
    drop_useless_columns(train_data, useless_columns)
    drop_useless_columns(test_data, useless_columns)
    log.msg('dropping useless columns', columns=useless_columns)

    dirty_columns: List[str] = ['Geography', 'Age', 'HasCrCard', 'IsActiveMember']
    drop_nan_rows(train_data, dirty_columns)
    drop_nan_rows(test_data, dirty_columns)
    log.msg('dropping NaN rows', columns=dirty_columns)

    column_mappings: Dict[str, Dict[Any, int]] = {
        'Geography': {
            'France': 0,
            'Spain': 1,
            'Germany': 2,
        },
        'Gender': {
            'Male': 0,
            'Female': 1,
        }
    }
    make_enum_columns_be_numeric(train_data, column_mappings)
    make_enum_columns_be_numeric(test_data, column_mappings)
    log.msg('making enum columns be numeric', columns=column_mappings.keys())
