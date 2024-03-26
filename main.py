from typing import List

from imblearn.over_sampling import SMOTE
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    log = structlog.get_logger()

    data: DataFrame = load_data("data/train.csv")
    log.msg('load train data', total=len(data))

    data = preprocess(data)
    log.msg('data preprocess done')

    X = data.drop('Exited', axis=1)
    y = data['Exited']

    X, y = SMOTE().fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # --------------------------

    data: DataFrame = load_data("data/test.csv")
    log.msg('load test data', total=len(data))

    data = preprocess(data)
    log.msg('data preprocess done')

    X = data

    X = scaler.transform(X)

    pred = model.predict(X)

    with open('522023330089.txt', 'w') as f:
        for line in pred:
            f.write(str(line) + '\n')
