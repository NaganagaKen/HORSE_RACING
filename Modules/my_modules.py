import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator


def common_process(df):
    df_copy = df.copy()

    # error_codeが1（出走取消）のデータは除外する
    df_copy = df_copy[df_copy.error_code != 1]

    # fold用レースidを作成
    df_copy["id_for_fold"] = df_copy["race_id"] // 100 # 下二桁を捨てる
    df_copy["id_for_fold"] = df_copy["id_for_fold"].astype("category")

    # コースの情報を記載（平地・障害、芝・ダート）
    course_type_dict = {
        10: '平地芝',
        11: '平地芝',
        12: '平地芝',
        17: '平地芝',
        18: '平地芝',
        21: '平地芝',
        23: '平地ダート',
        24: '平地ダート',
        52: '障害芝ダート',
        54: '障害芝',
        55: '障害芝',
        56: '障害芝',
        57: '障害芝'
    }
    df_copy["race_type"] = df_copy["track_code"].replace(course_type_dict)

    # 時系列順に並び替え
    place_dict = {
        "東京": 1,
        "中山": 2,
        "阪神": 3,
        "中京": 4,
        "京都": 5,
        "新潟": 6,
        "小倉": 7,
        "福島": 8,
        "札幌": 9,
        "函館": 10
        }
    df_copy["year"] += 2000
    df_copy["place_num"] = df_copy["place"].replace(place_dict).astype(int)
    df_copy["datetime"] = df_copy["year"].astype(str) + \
        df_copy["month"].astype(str).str.zfill(2) + \
        df_copy["day"].astype(str).str.zfill(2) + \
        df_copy["race_num"].astype(str).str.zfill(2) + \
        df_copy["place_num"].astype(str).str.zfill(2)
    df_copy["datetime"] = pd.to_datetime(df_copy["datetime"], format="%Y%m%d%H%M")
    df_copy = df_copy.sort_values("datetime")
    df_copy = df_copy.drop(["place_num"], axis=1)

    # horse_N = 5,6,7 は少ないので、8頭立てにまとめる
    df_copy["horse_N"] = df_copy["horse_N"].replace({5:8, 6:8, 7:8})

    # 必要ない特徴量は削除
    df_copy = df_copy.drop(["race_id"], axis=1)

    # とりあえず平地レースだけを使用
    df_copy = df_copy[(df_copy.race_type == "平地芝") | (df_copy.race_type == "平地ダート")]

    # カテゴリを示す数値列をカテゴリ列に変換
    to_category = ["jockey_id", "horse_N", "class_code", "track_code", "age_code", "weight_code"]
    for col in to_category:
        df_copy[col] = df_copy[col].astype("object")

    # ターゲット変数の作成
    df_copy["target"] = df_copy["rank"].apply(lambda x: 1 if x == 1 else 0)

    return df_copy


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
        s_groups = pd.Series(groups)

        for i in range(self.n_splits):
            train_end   = (i+1) * test_size
            train_groups = unique_groups[:train_end]

            test_start  = train_end
            if i < self.n_splits - 1:
                test_end    = test_start + test_size
                test_groups  = unique_groups[test_start:test_end]
            else :
                test_groups = unique_groups[test_start:]

            train_idx = np.where(s_groups.isin(train_groups))[0]
            test_idx  = np.where(s_groups.isin(test_groups))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def is_higher_than_odds(df, odds_series) :
    df["odds"] = odds_series
    df["is_higher_than_odds"] = (1/df["first_prize_prob"]) < df["odds"]
    print("the rate of higher than odds: ", df["is_higher_than_odds"].mean())

    return df