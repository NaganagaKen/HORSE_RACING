import numpy as np
import pandas as pd
from IPython.display import display
from trueskill import TrueSkill
from collections import defaultdict


# 勝率予測用特徴量エンジニアリング関数
def feature_engineering(df_to_copy, feature_col_to_copy=None):
    if feature_col_to_copy == None :
        # ブリンカーはあまり重要ではなさそうなので入れない
        feature_col_to_copy = ["waku_num", "horse_num", "sex", "age", "basis_weight", "weight", "inc_dec"]
    feature_col = feature_col_to_copy.copy()
    df = df_to_copy.copy()

    # 直近3レースの結果とその平均, 過去全てのレースの記録の平均を追加（PCIとRPCIはあまり重要ではなさそう）
    last_race_col = ["weight", "inc_dec", "last_3F_time", "Ave_3F"]
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

    # 過去その馬の全てのレースの1着率と複勝率
    #　シャープが3つついている列は、特徴量重要度の上位100位以内の列
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df,cols=["horse"]) ###

    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist"]) 
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["track_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["field_type"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["turn_type"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["weather"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["state"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["place"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["corner_num"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["class_code"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["basis_weight"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["age_code"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["weight_code"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["jockey_id", "field_type"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["weather", "state"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist", "corner_num"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist", "track_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist", "class_code"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["place", "field_type", "dist"])

    
    # 過去他の馬も含む全レースで同条件でのレースの1着の確率
    # dist, field_type, place, race_type, corner_num系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["dist", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["field_type", "place", "waku"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["race_type", "waku"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "waku"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["corner_num", "field_type", "waku"])

    # jockey_id系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "place"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "place", "dist"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "field_type"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "class_code", "field_type"]) ###

    # jockey_id-turn_type系
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "field_type"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "place", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "field_type", "waku"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["jockey_id", "turn_type", "dist", "place", "waku"])

    # trainer_id系(turn-typeと一緒に使うのは効果なし）)
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "race_type", "waku"])
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "field_type"]) ###
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df, cols=["trainer_id", "class_code", "race_type", "waku"])


    # 最後にまとめてdict_for_dfをdfにくっつける
    processed_df = pd.DataFrame(dict_for_df)
    df = pd.concat([df, processed_df], axis=1)


    # その他特徴量を追加
    # weightに関する特徴量
    # weightは300kg以下の馬がいないことからこのようにした。
    df["basis_weight_per_weight"] = df["basis_weight"] / df["weight"].clip(lower=300) * 100 # 斤量/馬体重（％） ###
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
    df = df.drop(["last_race_date", "interval"], axis=1)
    feature_col.append("interval_day")


    # TrueSkillの計算
    # horse
    print("calculating trueskill of horse is processing")
    df, feature_col = calc_trueskill_common(df, feature_col, "horse", "horse") ###
    # jockey
    print("calculating trueskill of jockey is processing")
    df, feature_col = calc_trueskill_common(df, feature_col, "jockey_id", "jockey") ###

    # horse ×　jockey
    df["HorseTrueSkill_times_JockeyTrueSkill"] = df["horse_TrueSkill"] * df["jockey_TrueSkill"]
    feature_col.append("HorseTrueSkill_times_JockeyTrueSkill")

        
    # 過去に特定グループ内のレーティングの平均がいくつか計算する関数
    # horse_TrueSkill
    # father系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "dist"], target_name="horse_TrueSkill") # 後でビン分割した距離も追加する予定
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "race_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "mother"], target_name="horse_TrueSkill")

    # mother系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "dist"], target_name="horse_TrueSkill") # 後でビン分割した距離も追加する予定
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "race_type"], target_name="horse_TrueSkill")

    # broodmaresire系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "dist"], target_name="horse_TrueSkill") # 後でビン分割した距離も追加する予定
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "race_type"], target_name="horse_TrueSkill")

    # father x broodmaresire系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "dist"], target_name="horse_TrueSkill") # 後でビン分割した距離も追加する予定
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "race_type"], target_name="horse_TrueSkill")

    # 後でjockey_id, trainer_idも追加予定(waku系はgrouped_winning_rateで対応)


    # 過去オッズの追加
    df, feature_col = merge_last_N_odds(df, feature_col)

    # 数値列全体を正規化（std=1とする)
    num_col = df[feature_col].select_dtypes(include=["number"]).columns.tolist()
    grouped_mean = df.groupby("id_for_fold", observed=True)[num_col].transform("mean")
    grouped_std = df.groupby("id_for_fold", observed=True)[num_col].transform("std")
    df[num_col] = (df[num_col] - grouped_mean) / grouped_std


    # ランキング特徴量
    group = df.groupby(["id_for_fold"], observed=True)
    ranking_col = ["horse_TrueSkill", "jockey_TrueSkill", "HorseTrueSkill_times_JockeyTrueSkill", "pre_win_odds_20"]
    
    for col in ranking_col:
        df[f"{col}_ranking"] = group[col].rank(ascending=False, method="min")
        feature_col.append(f"{col}_ranking")

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


# 過去に特定グループ内のレーティングの平均がいくつか計算する関数
# 収束が速いので、とりあえずTrueSkillのみで計算を行う
def calc_grouped_rating(df_to_copy, feature_col_to_copy, cols=None, target_name=None):
    # cols: List ... グループを決めるための特徴量のリスト
    # target_name: str ... どのレーティングのスコアを平均するかを指定する文字列
    if cols == None:
        raise ValueError("cols must be selected")
    elif target_name == None or type(target_name) != str:
        raise ValueError("target_col must be str")

    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy()
    target_col = target_name + "_after_racing"
    grouped1 = df.groupby(cols, observed=True)
    grouped2 = df.groupby(["id_for_fold", *cols], observed=True)

    # 同じ条件で1着になるの確率を計算
    bunsi1 = grouped1[target_col].cumsum() - grouped2[target_col].cumsum()
    bunbo1 = grouped1[target_col].cumcount() - grouped2[target_col].cumcount()

    feature_name = f"mean_{target_name}_in_group_" + "_".join(cols)
    feature_col.append(feature_name)
    df[feature_name] = bunsi1 / bunbo1.replace(0, np.nan)

    return df, feature_col
    

# 共通化されたTrueSkill計算関数
def calc_trueskill_common(df_to_copy, feature_col, target_col, prefix):
    """
    df : DataFrame
    feature_col : list
        特徴量リスト
    target_col : str
        対象となる列名（"horse" や "jockey_id"）
    prefix : str
        特徴量名の接頭辞（"horse" や "jockey"）
    """
    df = df_to_copy.copy()
    feature_col = feature_col.copy()

    CONFIDENCE_MULTIPLIER = 3 # 何シグマ範囲でTrueSkillのmin, maxを計算するかを定める

    df[f"{prefix}_TrueSkill"] = np.nan
    df[f"{prefix}_TrueSkill_sigma"] = np.nan
    df[f"{prefix}_TrueSkill_min"] = np.nan
    df[f"{prefix}_TrueSkill_max"] = np.nan
    df[f"{prefix}_TrueSkill_after_racing"] = np.nan # レース後のレーティング（特徴量には加えない、ターゲットエンコーディングに使用）

    feature_col.append(f"{prefix}_TrueSkill")
    feature_col.append(f"{prefix}_TrueSkill_sigma")
    feature_col.append(f"{prefix}_TrueSkill_min")
    feature_col.append(f"{prefix}_TrueSkill_max")

    env = TrueSkill(draw_probability=0.0)
    ratings = defaultdict(lambda: env.create_rating())

    grouped = df.groupby("id_for_fold", observed=True)

    for id, group in grouped:
        race_data = group[group["error_code"] == 0].copy()

        target_list = race_data[target_col].tolist()
        race_ratings = [[ratings[target]] for target in target_list]

        all_target_list = group[target_col].tolist()
        mu_array = [float(ratings[target].mu) for target in all_target_list]
        sigma_array = [float(ratings[target].sigma) for target in all_target_list]
        mask = (df["id_for_fold"] == id) & (df[target_col].isin(all_target_list))

        df.loc[mask, f"{prefix}_TrueSkill"] = mu_array
        df.loc[mask, f"{prefix}_TrueSkill_sigma"] = sigma_array
        df.loc[mask, f"{prefix}_TrueSkill_min"] = np.array(mu_array) - np.array(sigma_array) * CONFIDENCE_MULTIPLIER
        df.loc[mask, f"{prefix}_TrueSkill_max"] = np.array(mu_array) + np.array(sigma_array) * CONFIDENCE_MULTIPLIER

        ranks = race_data["rank"].tolist()
        new_ratings = env.rate(race_ratings, ranks=ranks)

        for target, new_group in zip(target_list, new_ratings):
            ratings[target] = new_group[0]

        # レース後のレートを埋め込み
        mu_array_after_racing = [float(ratings[target].mu) for target in all_target_list]
        df.loc[mask, f"{prefix}_TrueSkill_after_racing"] = mu_array_after_racing

    return df.copy(), feature_col



# オッズデータと結合する関数
def merge_last_N_odds(df, feature_col):
    # 過学習抑制の為にオッズ情報をなるべく加えない（5分前オッズのみを加える）。
    odds_df = pd.read_csv("../Data/Time_Series_Odds_win_odds.csv", encoding="shift-jis")
    odds_df_selected =  odds_df[["race_id", "pre_win_odds_20"]]

    new_feature_col = feature_col + ["pre_win_odds_20"]

    df = pd.merge(left=df, right=odds_df_selected, how="left", on=["race_id"])

    return df, new_feature_col