import numpy as np
import pandas as pd
from IPython.display import display
from trueskill import TrueSkill
from collections import defaultdict


# 勝率予測用特徴量エンジニアリング関数
def feature_engineering(df_to_copy, feature_col_to_copy=None):
    if feature_col_to_copy == None :
        feature_col_to_copy = ["waku_num", "horse_num", "sex", "age", "basis_weight", "blinker", "weight", "inc_dec"]
    feature_col = feature_col_to_copy.copy()
    df = df_to_copy.copy()

    # 直近3レースの結果とその平均, 過去全てのレースの記録の平均を追加
    last_race_col = ["weight", "inc_dec", "last_3F_time", "Ave_3F", "PCI", "RPCI"]
    for col in last_race_col:
        grouped = df.groupby("horse", observed=True)[col]
        for i in range(1, 4):
            # 過去1-3レースの結果を追加
            colname = f"{col}_last_{i}"
            df[colname] = grouped.shift(1)
            feature_col.append(colname)
        
        # 過去3レース分の結果の平均を追加
        df[f"{col}_mean_last_1_to_3"] = df[[f"{col}_last_{i}" for i in range(1, 4)]].mean(axis=1, skipna=True)
        feature_col.append(f"{col}_mean_last_1_to_3")

        # 過去全レース文の特徴量を追加
        cumsum = grouped.cumsum()
        count = grouped.cumcount()
        df[f"{col}_mean_all"] = (cumsum - df[col]) / count.replace(0, np.nan)
        feature_col.append(f"{col}_mean_all")


    # 特徴量を入れておくための辞書(fragment防止)
    dict_for_df = dict()

    # 過去その馬の全てのレースの1着率
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df,cols=["horse"])

    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["track_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["field_type"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["turn_type"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["weather"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["state"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["place"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["corner_num"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["class_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["basis_weight"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["age_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["weight_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id", "class_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id", "place"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id", "dist"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id", "field_type"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id", "place", "dist"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id", "place", "field_type", "dist"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["weather", "state"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist", "corner_num"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist", "track_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist", "class_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["place", "field_type", "dist"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["place", "field_type", "dist", "class_code"])

    # 過去他の馬も含む全レースで同条件でのレースの1着の確率
    # dist, field_type, place, race_type, corner_num系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["place", "horse_num"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "dist", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "dist", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "place", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "place", "horse_num"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["race_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["race_type", "horse_num"])
    
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "dist", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "dist", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "place", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "field_type", "horse_num"])



    # leg系(リーク情報なので一旦停止)
    '''
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "dist"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "place"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "field_type"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "place", "field_type"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "place", "dist"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "dist", "field_type"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "race_type"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "race_type", "waku"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "race_type", "waku_num"])
    dict_for_df, feature_col = grouped_leg_winning_rate(df, feature_col, dict_for_df, cols=["leg", "race_type", "horse_num"])
    '''

    # jockey_id系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "place", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "field_type", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "field_type", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "race_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "race_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "race_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "race_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "race_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "race_type", "horse_num"])

    # jockey_id-turn_type系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "field_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "place", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "field_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "field_type", "horse_num"])


    # trainer_id系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "place", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "field_type", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "field_type", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "race_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "race_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "race_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "race_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "race_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "race_type", "horse_num"])

    # trainer_id-turn_type系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "field_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "place", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "dist", "field_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "turn_type", "place", "field_type", "horse_num"])


    #jokey_id & trainer_id系


    # mother系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "turn_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "race_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "track_code"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "class_code"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "corner_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "horse_num"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "place", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "place", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "dist", "field_type"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "place", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "place", "dist", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "place", "dist", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "dist", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "dist", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "dist", "field_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "field_type", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "field_type", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "field_type", "place", "horse_num"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "race_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["mother", "race_type", "horse_num"])
        

    # father系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "place"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "turn_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "race_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "track_code"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "class_code"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "corner_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "horse_num"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "place", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "place", "field_type"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "dist", "field_type"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "place", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "place", "dist", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "place", "dist", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "dist", "field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "dist", "field_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "dist", "field_type", "horse_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "field_type", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "field_type", "place", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "field_type", "place", "horse_num"])

    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "race_type", "waku_num"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["father", "race_type", "horse_num"])
    
    # broodmare_sire系

    # broodmare_sire_type系

    # horse_color系



    # 最後にまとめてdict_for_dfをdfにくっつける
    processed_df = pd.DataFrame(dict_for_df)
    df = pd.concat([df, processed_df], axis=1)


    # その他特徴量を追加
    # weightに関する特徴量
    # weightは300kg以下の馬がいないことからこのようにした。
    df["basis_weight_per_weight"] = df["basis_weight"] / df["weight"].clip(lower=300) * 100 # 斤量/馬体重（％）
    feature_col.append("basis_weight_per_weight")
    df["basis_weight_plus_weight"] = df["basis_weight"] + df["weight"] # 斤量＋馬体重
    feature_col.append("basis_weight_plus_weight")
    df["inc_dec_rate"] = df["inc_dec"] / df["weight"].clip(lower=300) * 100 # 増減/馬体重（％）
    feature_col.append("inc_dec_rate")

    # 生涯獲得賞金
    df["lifetime_prize"] = df.groupby("horse", observed=True)["prize"].cumsum() - df["prize"]
    feature_col.append("lifetime_prize")

    # 生涯獲得賞金 / 今まで出走したレース
    df["lifetime_prize_per_race"] = df["lifetime_prize"] / df.groupby("horse", observed=True)["prize"].cumcount().replace(np.nan, 0)
    feature_col.append("lifetime_prize_per_race")
    
    # 前回と同じfield_typeかどうか
    df["last_field_type"] = df.groupby(["horse"], observed=True)["field_type"].shift(1)
    feature_name = "is_same_field_type_as_last"
    df[feature_name] =  df["field_type"] == df["last_field_type"]
    df[feature_name] = df[feature_name].astype("category")
    df = df.drop(["last_field_type"], axis=1)
    feature_col.append(feature_name)

    # 前回と同じクラスか
    df["last_class_code"] = df.groupby(["horse"], observed=True)["class_code"].shift(1)
    feature_name = "is_same_class_code_as_last"
    df[feature_name] = df["class_code"] == df["last_class_code"]
    df[feature_name] = df[feature_name].astype("category")
    df = df.drop(["last_class_code"], axis=1)
    feature_col.append(feature_name)

    # 前回と同じジョッキーか
    df["last_jockey"] = df.groupby(["horse"], observed=True)["jockey_id"].shift(1)
    feature_name = "is_same_jockey_as_last"
    df[feature_name] = df["jockey_id"] == df["last_jockey"]
    df[feature_name] = df[feature_name].astype("category")
    df = df.drop(["last_jockey"], axis=1)
    feature_col.append(feature_name)

    # 中何日か
    df["last_race_date"] = df.groupby("horse", observed=True)["datetime"].shift(1)
    df["interval"] = df["datetime"] - df["last_race_date"]
    df["interval_day"] = df["interval"].dt.days
    df["interval_week"] = df["interval_day"] // 7
    df = df.drop(["last_race_date", "interval"], axis=1)
    feature_col.append("interval_day")
    feature_col.append("interval_week")

    # 今回が一番位の高いレースか
    df["class_code"] = df["class_code"].astype(int) # categoryなので、一度intに変換（cummaxが使えないため）
    df["is_highest_class"] = df.groupby("horse", observed=True)["class_code"].cummax() == df["class_code"]
    feature_col.append("is_highest_class")
    df["class_code"] = df["class_code"].astype("category") # categoryに戻す


    # TrueSkillの計算
    # horse
    print("calculating trueskill of horse is processing")
    df = calc_trueskill_horse(df)
    feature_col.append("horse_TrueSkill")
    # jockey
    print("calculating trueskill of jockey is processing")
    df = calc_trueskill_jockey(df)
    feature_col.append("jockey_TrueSkill")
    # horse ×　jockey
    df["HorseTrueSkill_times_JockeyTrueSkill"] = df["horse_TrueSkill"] * df["jockey_TrueSkill"]
    feature_col.append("HorseTrueSkill_times_JockeyTrueSkill")
    # horse + jockey
    df["HorseTrueSkill_plus_JockeyTrueSkill"] = df["horse_TrueSkill"] + df["jockey_TrueSkill"]
    feature_col.append("HorseTrueSkill_plus_JockeyTrueSkill")


    # 最後に全体を正規化（std=1とする)
    num_col = df[feature_col].select_dtypes(include=["number"]).columns.tolist()
    grouped_mean = df.groupby("id_for_fold", observed=True)[num_col].transform("mean")
    grouped_std = df.groupby("id_for_fold", observed=True)[num_col].transform("std")
    df[num_col] = (df[num_col] - grouped_mean) / grouped_std

    # 後でランキング化とかも付ける予定

    # dfを表示
    print(feature_col)
    display(df.tail())

    return df, feature_col


# 馬でグループ化したtarget-encodingをする関数
def grouped_horse_winning_rate(df_to_copy, feature_col_to_copy, cols=None):
    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy()

    if cols == None :
        print("Error: please select cols")
        return
    
    # 1着の確率で計算
    grouped = df.groupby(["horse", *cols], observed=True)["target"]
    cumsum = grouped.cumsum()
    count = grouped.cumcount()
    feature_name = "horse_win_rate_" + "_".join(cols)
    df[feature_name] = (cumsum-df["target"]) / count.replace(0, np.nan)

    feature_col.append(feature_name)

    # 1-3着の確率で計算
    grouped = df.groupby(["horse", *cols], observed=True)["target3"]
    cumsum = grouped.cumsum()
    count = grouped.cumcount()
    feature_name = "horse_win_rate3_" + "_".join(cols)
    df[feature_name] = (cumsum-df["target3"]) / count.replace(0, np.nan)

    feature_col.append(feature_name)

    return df, feature_col


# 過去全てのレースでグループ化したtarget-encodingをする関数
def grouped_winning_rate(df_to_copy, feature_col_to_copy, dict_for_df, cols):
    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy()
    grouped1 = df.groupby(cols, observed=True)
    grouped2 = df.groupby(["id_for_fold", *cols], observed=True)

    # 同じ条件で1着になるの確率を計算
    bunsi1 = grouped1["target"].cumsum() - grouped2["target"].cumsum()
    bunbo1 = grouped1["target"].cumcount() - grouped2["target"].cumcount()

    feature_name = "all_win_rate_" + "_".join(cols)
    feature_col.append(feature_name)
    dict_for_df[feature_name] = bunsi1 / bunbo1.replace(0, np.nan)

    # 同じ条件で1-3着になるの確率を計算
    bunsi3 = grouped1["target3"].cumsum() - grouped2["target3"].cumsum()
    bunbo3 = grouped1["target3"].cumcount() - grouped2["target3"].cumcount()

    feature_name3 = "all_win_rate3_" + "_".join(cols)
    feature_col.append(feature_name3)
    dict_for_df[feature_name3] = bunsi3 / bunbo3.replace(0, np.nan)


    return dict_for_df, feature_col


# 各馬のTrueSkillを計算する関数
def calc_trueskill_horse(df):
    df = df.copy()
    df["horse_TrueSkill"] = np.nan

    env = TrueSkill(draw_probability=0.0) # TrueSkill環境
    ratings = defaultdict(lambda:env.create_rating()) # 全馬のレートが入っている辞書

    grouped = df.groupby("id_for_fold", observed=True)

    for id, group in grouped:
        race_data = group[group["error_code"] == 0].copy()

        horse_list = race_data["horse"].tolist()
        race_ratings = [[ratings[horse]] for horse in horse_list]

        # 各馬のレーティングを埋め込み
        # error_codeが0ではない馬（異常終了）は一つ前のレースのデータを埋め込む
        all_horse_list = group["horse"].tolist()
        mu_array = [ratings[horse].mu for horse in all_horse_list]
        mask = (df["id_for_fold"] == id) & (df["horse"].isin(all_horse_list))
        df.loc[mask, "horse_TrueSkill"] = mu_array

        # レーティングの更新
        ranks = race_data["rank"].tolist() # レースの結果
        new_ratings = env.rate(race_ratings, ranks=ranks)

        for horse, new_group in zip(horse_list, new_ratings):
            ratings[horse] = new_group[0]


    return df


# 各ジョッキーのTrueSkillを計算する関数
def calc_trueskill_jockey(df):
    df = df.copy()
    df["jockey_TrueSkill"] = np.nan

    env = TrueSkill(draw_probability=0.0) # TrueSkill環境
    ratings = defaultdict(lambda:env.create_rating()) # 全馬のレートが入っている辞書

    grouped = df.groupby("id_for_fold", observed=True)

    for id, group in grouped:    
        race_data = group[group["error_code"] == 0].copy()

        jockey_list = race_data["jockey_id"].tolist()
        race_ratings = [[ratings[jockey]] for jockey in jockey_list]

        # 各馬のレーティングを埋め込み
        # error_codeが0ではない馬（異常終了）は一つ前のレースのデータを埋め込む
        all_jockey_list = group["jockey_id"].tolist()
        mu_array = [ratings[jockey].mu for jockey in all_jockey_list]
        mask = (df["id_for_fold"] == id) & (df["jockey_id"].isin(all_jockey_list))
        df.loc[mask, "jockey_TrueSkill"] = mu_array

        # レーティングの更新
        ranks = race_data["rank"].tolist() # レースの結果
        new_ratings = env.rate(race_ratings, ranks=ranks)

        for jockey, new_group in zip(jockey_list, new_ratings):
            ratings[jockey] = new_group[0]


    return df