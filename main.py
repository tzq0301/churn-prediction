import argparse
from typing import List

import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import structlog


def load_data(filepath: str) -> DataFrame:
    return pd.read_csv(filepath, index_col=0)


def drop_useless_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    for column in columns:
        if column in df.columns:
            df.drop(column, axis=1, inplace=True)
    return df


def impute_nan_rows(df: DataFrame, medians: List[str], most_frequents: List[str]) -> DataFrame:
    for column in medians:
        median_imputer = SimpleImputer(strategy='median')
        df[[column]] = median_imputer.fit_transform(df[[column]])
    for column in most_frequents:
        median_imputer = SimpleImputer(strategy='most_frequent')
        df[[column]] = median_imputer.fit_transform(df[[column]])
    return df


def one_hot_encoding(df: DataFrame, columns: List[str]) -> DataFrame:
    for column in columns:
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        df = df.reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        df = pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
    return df


def preprocess(df: DataFrame) -> DataFrame:
    df = drop_useless_columns(df, ['Surname', 'CustomerId'])
    df = impute_nan_rows(df, medians=['Age'], most_frequents=['HasCrCard', 'IsActiveMember', 'Geography'])
    df = one_hot_encoding(df, columns=['Gender', 'Geography'])
    return df


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--filepath", type=str, required=True, help="path to data file")
    args_parser.add_argument("--mode", type=str, required=True, choices=['train', 'predict'])
    args_parser.add_argument("--output", type=str, help="output file in predict mode")
    args = args_parser.parse_args()

    log = structlog.get_logger()

    data: DataFrame = load_data(args.filepath)
    log.msg('load data', mode=args.mode, filepath=args.filepath, total=len(data))

    data = preprocess(data)
    log.msg('data preprocess done')
