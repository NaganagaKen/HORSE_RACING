import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator


def common_process(df_to_copy):
    df = df_to_copy.copy()

    # error_codeが1（出走取消）のデータは除外する
    df = df[df.error_code != 1]

    # fold用レースidを作成
    df["id_for_fold"] = df["race_id"] // 100 # 下二桁を捨てる
    df["id_for_fold"] = df["id_for_fold"].astype("category")

    # コースの情報を記載（平地・障害、芝・ダート）
    field_type_dict = {
    10: "芝", 11: "芝", 12: "芝", 13: "芝", 14: "芝", 15: "芝", 16: "芝",
    17: "芝", 18: "芝", 19: "芝", 20: "芝", 21: "芝", 22: "芝",
    23: "ダート", 24: "ダート", 25: "ダート", 26: "ダート", 29: "ダート",
    27: "サンド", 28: "サンド",
    51: "芝", 52: "芝ダート", 53: "芝", 54: "芝", 55: "芝", 56: "芝", 57: "芝", 58: "芝", 59: "芝"
    }

    df["field_type"] = df["track_code"].replace(field_type_dict)
    
    # レースのタイプ
    race_type_dict = {
    10: "平地", 11: "平地", 12: "平地", 13: "平地", 14: "平地", 15: "平地", 16: "平地",
    17: "平地", 18: "平地", 19: "平地", 20: "平地", 21: "平地", 22: "平地",
    23: "平地", 24: "平地", 25: "平地", 26: "平地", 27: "平地", 28: "平地", 29: "平地",
    51: "障害", 52: "障害", 53: "障害", 54: "障害", 55: "障害", 56: "障害", 57: "障害", 58: "障害", 59: "障害"
    }

    df["flat_or_jump"] = df["track_code"].replace(race_type_dict)

    # 右回り(R)・左回り(L)・直線(S)・障害レース(N)かを判定する
    # 障害レースはかなり適当だから注意。後で直す
    turn_type_dict = {
    10: "S", 29: "S",  # 直線
    11: "L", 12: "L", 13: "L", 14: "L", 15: "L", 16: "L", 23: "L", 25: "L", 27: "L", 53: "L",
    17: "R", 18: "R", 19: "R", 20: "R", 21: "R", 22: "R", 24: "R", 26: "R", 28: "R",
    51: "S",  # 襷コースは直線扱い（実際はジグザグだが一種の直線）
    52: "L", 54: "L", 55: "L", 56: "L", 57: "L", 58: "L", 59: "L"
    }
    df["turn_type"] = df["track_code"].replace(turn_type_dict)


    # レースを識別するための会場・（芝orダート）・距離という特徴量を作成
    # 「阪神芝1600」みたいな特徴量ができる
    df["race_type"] = df["place"] + df["field_type"] + df["dist"].astype(str) 

    #　内枠か外枠かを表す特徴量wakuを作成
    df["waku"] = df["waku_num"].apply(lambda x: "inner" if 1<=x<=4 else "outer")


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
    df["year"] += 2000
    df["place_num"] = df["place"].replace(place_dict).astype(int)
    df["datetime"] = df["year"].astype(str) + \
        df["month"].astype(str).str.zfill(2) + \
        df["day"].astype(str).str.zfill(2) + \
        df["race_num"].astype(str).str.zfill(2) + \
        df["place_num"].astype(str).str.zfill(2)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y%m%d%H%M")
    df = df.sort_values("datetime")
    df = df.drop(["place_num"], axis=1)

    # 必要ない特徴量は削除
    df = df.drop(["race_id"], axis=1)

    # とりあえず平地レースだけを使用
    df = df[df["flat_or_jump"] == "平地"]

    # カテゴリを示す数値列をカテゴリ列に変換
    to_category = ["jockey_id", "horse_N", "class_code", "track_code", "age_code", "weight_code"]
    for col in to_category:
        df[col] = df[col].astype("object")

    # ターゲット変数の作成
    df["target"] = df["rank"].apply(lambda x: 1 if x == 1 else 0)
    df["target3"] = df["rank"].apply(lambda x: 1 if 1 <= x <= 3 else 0)

    return df


# 競馬用の時系列かつグループ単位でバリデーションを行うsplitter
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