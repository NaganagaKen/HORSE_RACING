import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from lightgbm import LGBMClassifier


def common_process(df):
    # データリークの恐れのある特徴量の削除
    df = df.drop(['time_diff', 'time', 'corner1_rank', 'corner2_rank', 'corner3_rank', 'corner4_rank', 'last_3F_time', 'last_3F_rank', 'Ave_3F', 'PCI', 'last_3F_time_diff', "win_odds", "prize"], axis=1)
    
    # fold用レースidを作成
    df["id_for_fold"] = df["race_id"] // 100 # 下二桁を捨てる
    df["id_for_fold"] = df["id_for_fold"].astype("category")

    # 時系列順に並び替え
    df["year"] += 2000
    df["datetime"] = df["year"].astype(str) + \
        df["month"].astype(str).str.zfill(2) + \
        df["day"].astype(str).str.zfill(2) + \
        df["times"].astype(str).str.zfill(2) + \
        df["race_num"].astype(str).str.zfill(2)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M")
    df = df.sort_values("datetime")
    df = df.drop(["datetime"], axis=1)

    # 必要ない特徴量は削除
    df = df.drop(["race_id"], axis=1)

    # ターゲット変数の作成
    df["target"] = df["rank"].apply(lambda x: 1 if x == 1 else 0)
    df = df.drop(["rank"], axis=1)

    return df


class GroupTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided")

        unique_groups = pd.Series(groups).drop_duplicates().values
        n_groups = len(unique_groups)

        if self.n_splits >= n_groups:
            raise ValueError("n_splits must be < n_groups")

        test_size = n_groups // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end   = (i+1) * test_size
            train_groups = unique_groups[:train_end]

            test_start  = train_end
            if i < self.n_splits - 1: 
                test_end    = test_start + test_size
                test_groups  = unique_groups[test_start:test_end]
            else :
                test_groups = unique_groups[test_start:]
            
            train_idx = np.where(pd.Series(groups).isin(train_groups))[0]
            test_idx  = np.where(pd.Series(groups).isin(test_groups))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def simple_lightGBM(df):
    splitter = GroupTimeSeriesSplit(n_splits=5)

    cat_col = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_col:
        df[col] = df[col].astype("category")

    X, y = df.drop(["target"], axis=1), df["target"]
    oof_prob = np.full(len(df), np.nan)

    # lightGBMのパラメータを調整する
    params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 1000,
            "random_state": 42,
            "verbose": -1
        }

    for tr_idx, val_idx in splitter.split(X, groups=df["id_for_fold"]):
        X_train, y_train = X.iloc[tr_idx], y.iloc[tr_idx]
        X_test, y_test = X.iloc[val_idx], y.iloc[val_idx]


        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        oof_prob[val_idx] = y_pred_prob

    score = log_loss(y[~np.isnan(oof_prob)], oof_prob[~np.isnan(oof_prob)])
    print("logloss score: ", round(score, 10))

    final_model = LGBMClassifier(**params)
    final_model.fit(X, y)

    return final_model, oof_prob


def calculator_prob(df, prob):
    df["unnormalized_prob"] = prob
    sum_prob = df.groupby(["id_for_fold"])["unnormalized_prob"].sum()
    sum_prob = pd.DataFrame({"sum_prob": sum_prob})
    df = pd.merge(df, sum_prob, how="left", on="id_for_fold")
    df["first_prize_prob"] = df["unnormalized_prob"] / df["sum_prob"]

    df = df.drop(["unnormalized_prob", "sum_prob"], axis=1)
    return df


def is_higher_than_odds(df, odds_series) :
    df["odds"] = odds_series
    df["is_higher_than_odds"] = (1/df["first_prize_prob"]) < df["odds"]
    print("the rate of higher than odds: ", df["is_higher_than_odds"].mean())

    return df