import numpy as np
import pandas as pd
from IPython.display import display
from trueskill import TrueSkill
from itertools import combinations
from collections import defaultdict
from glicko2 import Player
from sklearn.preprocessing import PolynomialFeatures

pd.option_context(
        'display.max_info_rows', None,     # 行しきい値を無制限
        'display.max_info_columns', None
        )


# 勝率予測用特徴量エンジニアリング関数
def feature_engineering(df_to_copy, feature_col_to_copy=None, tansho_odds_path="../Data/tansho/tansho_2021_2025.csv"):
    if feature_col_to_copy == None :
        # ブリンカーはあまり重要ではなさそうなので入れない
        feature_col_to_copy = ["waku_num", "horse_num", "sex", "age", "basis_weight", "weight", "inc_dec"]
    if tansho_odds_path is None:
        raise ValueError("Error in merge_last_N_odds: tansho_odds_path must be specified")

    feature_col = feature_col_to_copy.copy()
    df = df_to_copy.copy()

    ranking_col = [] # ランキング化する特徴量の名前を入れるリスト（特に重要な特徴量はランキング化する）

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

    # 単変量特徴量を追加
    df["num_of_entries"] = df.groupby("horse", observed=True)["horse"].cumcount()
    feature_col.append("num_of_entries")

    # 過去選択された脚質の回数と確率を追加
    df, feature_col = calc_leg_cumsum(df, feature_col)
    # 過去の平均着順、平均コーナー通過順など、レース展開と結果の平均を埋め込む
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="rank", grouping_col=["horse"], feature_name="past_rank_mean")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="rank", grouping_col=["horse", "waku"], feature_name="past_rank_mean_grouped_waku")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="rank", grouping_col=["horse", "dist_type"], feature_name="past_rank_mean_grouped_dist_type")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="rank", grouping_col=["horse", "dist"], feature_name="past_rank_mean_grouped_dist")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="rank", grouping_col=["horse", "class_code"], feature_name="past_rank_mean_grouped_class_code")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="rank", grouping_col=["horse", "place"], feature_name="past_rank_mean_grouped_place")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="rank", grouping_col=["horse", "dist_type", "waku"], feature_name="past_rank_mean_grouped_dist_type_waku")
    for i in range(1, 5):
        df, feature_col = calc_mean_race_development(df, feature_col, target_col=f"corner{i}_rank", grouping_col=["horse"], 
                                                     feature_name=f"past_corner{i}_rank_mean")
        df, feature_col = calc_mean_race_development(df, feature_col, target_col=f"corner{i}_rank", grouping_col=["horse", "dist_type"], 
                                                     feature_name=f"past_corner{i}_rank_mean_grouped_dist_type")
        df, feature_col = calc_mean_race_development(df, feature_col, target_col=f"corner{i}_rank", grouping_col=["horse", "dist"], 
                                                     feature_name=f"past_corner{i}_rank_mean_grouped_dist")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="pop", grouping_col=["horse"], feature_name="past_pop_mean")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="pop", grouping_col=["horse", "dist"], feature_name="past_pop_mean_grouped_dist")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="pop", grouping_col=["horse", "dist_type"], feature_name="past_pop_mean_grouped_dist_type")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="pop", grouping_col=["horse", "place"], feature_name="past_pop_mean_grouped_place")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="pop", grouping_col=["horse", "class_code"], feature_name="past_pop_mean_grouped_class_code")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_rank", grouping_col=["horse"], feature_name="past_last_3F_rank_mean")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_rank", grouping_col=["horse", "dist"], feature_name="past_last_3F_rank_mean_grouped_dist")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_rank", grouping_col=["horse", "dist_type"], feature_name="past_last_3F_rank_mean_grouped_dist_type")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_rank", grouping_col=["horse", "dist_type", "waku"], feature_name="past_last_3F_rank_mean_grouped_dist_type_waku")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_time", grouping_col=["horse"], feature_name="past_last_3F_time_mean")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_time", grouping_col=["horse", "dist"], feature_name="past_last_3F_time_mean_grouped_dist")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_time", grouping_col=["horse", "dist_type"], feature_name="past_last_3F_time_mean_grouped_dist_type")
    df, feature_col = calc_mean_race_development(df, feature_col, target_col="last_3F_time", grouping_col=["horse", "dist_type", "waku"], feature_name="past_last_3F_time_mean_grouped_dist_type_waku")


    # 相互作用特徴量を追加
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

    # 中何日か
    df["last_race_date"] = df.groupby("horse", observed=True)["datetime"].shift(1)
    df["interval"] = df["datetime"] - df["last_race_date"]
    df["interval_day"] = df["interval"].dt.days
    df = df.drop(["last_race_date", "interval"], axis=1)
    feature_col.append("interval_day")


    # TrueSkillの計算
    print("calculating trueskill is in progress")
    df, feature_col = calc_trueskill_fast(df, feature_col, "horse", "horse") ###
    df, feature_col = calc_trueskill_fast(df, feature_col, "jockey_id", "jockey") ###

    # EloRatingの計算
    print("calculating EloRating is in progress")
    df, feature_col = calc_elo_rating_fast(df, feature_col, target_col="horse", prefix="horse")
    df, feature_col = calc_elo_rating_fast(df, feature_col, target_col="jockey_id", prefix="jockey")

    # Glicko2の計算
    print("calculating Glicko2 is in progress")
    df, feature_col = calc_glicko2_common(df, feature_col, target_col="horse", prefix="horse")
    #jockeyの部分は、なぜかエラーが出るので、gitからソースコードを引っ張ってきて、それを直接直そうと思う。
    #df, feature_col = calc_glicko2_common(df, feature_col, target_col="jockey_id", prefix="jockey") 

    # 各レートの上昇量を計算
    rating_diff_list = ['horse_TrueSkill', 'jockey_TrueSkill', 'horse_EloRating', 'jockey_EloRating', 'horse_Glicko2'] # 空白区切り
    for col in rating_diff_list:
        df, feature_col = calc_rating_diff(df, feature_col, target_col=col, prefix=col)


    # レーティングの相互作用特徴量を追加
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_list = ['horse_TrueSkill', 'horse_TrueSkill_min',
                        'horse_TrueSkill_max', 'jockey_TrueSkill',
                        'jockey_TrueSkill_min', 'jockey_TrueSkill_max', 'horse_EloRating', 
                        'jockey_EloRating', 'horse_Glicko2',
                        'horse_Glicko2_min', 'horse_Glicko2_max']
    poly_features = poly.fit_transform(df[poly_list])
    poly_features_name = poly.get_feature_names_out(poly_list)
    poly_features_df = pd.DataFrame(poly_features, columns=poly_features_name, index=df.index)

    # 15 個の元列だけ除外してから結合（PolynomialFeaturesは相互作用を計算しないものまで含めてしまう）
    interaction_cols = [c for c in poly_features_name if c not in poly_list]
    feature_col.extend(interaction_cols)
    ranking_col.extend(["jockey_TrueSkill horse_Glicko2", "jockey_TrueSkill_min horse_Glicko2", "jockey_TrueSkill_max horse_Glicko2"])
    df = pd.concat([df, poly_features_df[interaction_cols]], axis=1)

    print("poly calculated")


    # 過去に特定グループ内のレーティングの平均がいくつか計算する関数
    # horse_TrueSkill
    # father系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "age_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "race_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "mother"], target_name="horse_TrueSkill")

    # mother系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "age_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["mother", "race_type"], target_name="horse_TrueSkill")

    # broodmaresire系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "age_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["broodmare_sire", "race_type"], target_name="horse_TrueSkill")

    # father x broodmaresire系
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "age_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "state"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "state", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "corner_num"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "place", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "dist"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "dist_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "turn_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "field_type"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "jockey_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "trainer_id"], target_name="horse_TrueSkill")
    df, feature_col = calc_grouped_rating(df, feature_col, cols=["father", "broodmare_sire", "race_type"], target_name="horse_TrueSkill")

    # 後でjockey_id, trainer_idも追加予定(waku系はgrouped_winning_rateで対応)

    # horse_TrueSkillではなく、賞金によって適性を見る関数
    df, feature_col = calc_lifetime_prize_cumsum(df, feature_col, cols=["father"])
    df, feature_col = calc_lifetime_prize_cumsum(df, feature_col, cols=["broodmare_sire"])
    df, feature_col = calc_lifetime_prize_cumsum(df, feature_col, cols=["mother"])
    df, feature_col = calc_lifetime_prize_cumsum(df, feature_col, cols=["father", "mother"])
    df, feature_col = calc_lifetime_prize_cumsum(df, feature_col, cols=["father", "broodmare_sire"])

    # --- 後でもう少し追加予定 --- 

    print("calc grouped rating caluculated")

    # 過去その馬の全てのレースの1着率と複勝率
    # 特徴量を入れておくための辞書(fragment防止)
    dict_for_df = dict()
    
    #　シャープが3つついている列は、特徴量重要度の上位100位以内の列
    dict_for_df, feature_col = grouped_winning_rate(df, feature_col, dict_for_df,cols=["horse"]) ###
    
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist"]) 
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist_type"]) 
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
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist_type", "corner_num"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist_type", "track_code"])
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["dist_type", "class_code"]) ###
    df, feature_col = grouped_horse_winning_rate(df, feature_col, cols=["place", "field_type", "dist_type"])

    
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

    print("group_winning_rate_calculated")

    # 過去オッズの追加(gainが高すぎるので一度加えないでおく)
    df, feature_col = merge_last_N_odds(df, feature_col, tansho_odds_path=tansho_odds_path)
    ranking_col.append("tansho_odds_20")
    print("added last odds")

    ## ここから開始
    show_df_duplicate_columns(df)

    # 数値列全体を正規化（std=1とする)

    if df.columns.duplicated().any():
        print("★ df に重複列があります → 先頭を残して削除します")
        df = df.loc[:, ~df.columns.duplicated()]
        feature_col = [c for c in feature_col if c in df.columns]  # DF に残っている列だけ
        feature_col = list(dict.fromkeys(feature_col)) 

    num_col = df[feature_col].select_dtypes(include=["number"]).columns.unique()
    for col in num_col:
        if np.issubdtype(df[col].dtype, np.integer): # intではなくnp.integerを指定しないといけない
            df[col] = df[col].astype(np.float64)   
    
    grouped_mean = df.groupby("id_for_fold", observed=True)[num_col].transform("mean")
    grouped_std = df.groupby("id_for_fold", observed=True)[num_col].transform("std")
    df.loc[:, num_col] = ((df[num_col] - grouped_mean) / grouped_std).astype(np.float64)

    print("num_col are standardize")


    # ランキング特徴量
    group = df.groupby(["id_for_fold"], observed=True)
    
    for col in ranking_col:
        df[f"{col}_ranking"] = group[col].rank(ascending=False, method="min")
        feature_col.append(f"{col}_ranking")

    print("calculated rankings")

    # dfを表示
    print(feature_col)
    display(df.tail())

    return df, feature_col


# ------------------------------------ メイン関数ここまで -------------------------------------------------

def show_df_duplicate_columns(df):
    dup_cols = df.columns[df.columns.duplicated()].unique()
    if len(dup_cols):
        print("▼ DataFrame 内で重複している列:", list(dup_cols))


# 過去に選択された脚質を追加（回数+確率）
def calc_leg_cumsum(df_to_copy, feature_col_to_copy):
    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy()
    target_col = ['後方', '中団', '逃げ', '先行', 'ﾏｸﾘ'] # なぜか追込がいない... 

    df["num_of_entries"] = df.groupby("horse", observed=True)["horse"].cumcount()

    leg_dummy = pd.get_dummies(df["leg"], drop_first=False).astype(int)
    df = pd.concat([df, leg_dummy], axis=1)

    grouped1 = df.groupby("horse", observed=True)
    grouped2 = df.groupby(["id_for_fold", "horse"], observed=True)

    # 同じ条件で1着になるの確率を計算
    bunsi1 = grouped1[target_col].cumsum() - grouped2[target_col].cumsum()

    for col in target_col:
        feature_name1 = f"{col}_per_entries"
        feature_name2 = f"{col}_cumcount_past_racing"
        df[feature_name1] = bunsi1[col] / df["num_of_entries"].replace(0, np.nan) # 脚質の選択確率を追加
        feature_col.append(feature_name1)
        df[feature_name2] = bunsi1[col] # 脚質の選択回数を追加
        feature_col.append(feature_name2)

    return df, feature_col


# 馬でグループ化したtarget-encodingをする関数
def grouped_horse_winning_rate(df_to_copy, feature_col_to_copy, cols=None):
    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy()

    if cols == None :
        print("Error: please select cols")
        return
    
    # 1着の確率で計算
    grouped = df.groupby(["horse", *cols], observed=True)["is_1st_rank"]
    cumsum = grouped.cumsum()
    count = grouped.cumcount()
    feature_name = "horse_win_rate_" + "_".join(cols)
    df[feature_name] = (cumsum-df["is_1st_rank"]) / count.replace(0, np.nan)

    feature_col.append(feature_name)

    # 1-3着の確率で計算
    grouped = df.groupby(["horse", *cols], observed=True)["is_in_3rd_rank"]
    cumsum = grouped.cumsum()
    count = grouped.cumcount()
    feature_name = "horse_win_rate3_" + "_".join(cols)
    df[feature_name] = (cumsum-df["is_in_3rd_rank"]) / count.replace(0, np.nan)

    feature_col.append(feature_name)

    return df, feature_col


# 過去全てのレースでグループ化したtarget-encodingをする関数
def grouped_winning_rate(df_to_copy, feature_col_to_copy, dict_for_df, cols):
    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy()
    grouped1 = df.groupby(cols, observed=True)
    grouped2 = df.groupby(["id_for_fold", *cols], observed=True)

    # 同じ条件で1着になるの確率を計算
    bunsi1 = grouped1["is_1st_rank"].cumsum() - grouped2["is_1st_rank"].cumsum()
    bunbo1 = grouped1["is_1st_rank"].cumcount() - grouped2["is_1st_rank"].cumcount()

    feature_name = "all_win_rate_" + "_".join(cols)
    feature_col.append(feature_name)
    dict_for_df[feature_name] = bunsi1 / bunbo1.replace(0, np.nan)

    # 同じ条件で1-3着になるの確率を計算
    bunsi3 = grouped1["is_in_3rd_rank"].cumsum() - grouped2["is_in_3rd_rank"].cumsum()
    bunbo3 = grouped1["is_in_3rd_rank"].cumcount() - grouped2["is_in_3rd_rank"].cumcount()

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


# 同じ条件の馬がどの程度の賞金を獲得しているのかの平均値を計算する関数
# 主にリーディング情報を埋め込む関数
def calc_lifetime_prize_cumsum(df_to_copy, feature_col_to_copy, cols=None, target_name="prize"):
    # cols: List ... グループを決めるための特徴量のリスト
    # target_name: str ... どのレーティングのスコアを平均するかを指定する文字列
    if cols == None:
        raise ValueError("cols must be selected")
    elif target_name == None or type(target_name) != str:
        raise ValueError("target_col must be str")

    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy()
    target_col = target_name
    grouped1 = df.groupby(cols, observed=True)
    grouped2 = df.groupby(["id_for_fold", *cols], observed=True)

    # 同じ条件での累積和と平均値
    bunsi1 = grouped1[target_col].cumsum() - grouped2[target_col].cumsum()
    bunbo1 = grouped1[target_col].cumcount() - grouped2[target_col].cumcount()

    feature_name_prize = f"all_{target_name}_in_group_" + "_".join(cols)
    feature_name_prize_per_racing = f"all_{target_name}_per_racing_in_group_" + "_".join(cols)
    feature_col.append(feature_name_prize)
    feature_col.append(feature_name_prize_per_racing)

    df[feature_name_prize_per_racing] = bunsi1
    df[feature_name_prize] = bunsi1 / bunbo1.replace(0, np.nan)

    return df, feature_col


# リークとなり得る情報から特徴量を作成する関数
def calc_mean_race_development(df_to_copy, feature_col_to_copy, target_col=None, grouping_col=None, feature_name=None):
    if (target_col is None) or (grouping_col is None) or (feature_name is None):
        raise ValueError("target_col, grouping_col and prefix must be selected")
    
    df = df_to_copy.copy()
    feature_col = feature_col_to_copy.copy() 

    grouped1 = df.groupby(grouping_col, observed=True)
    grouped2 = df.groupby(["id_for_fold", *grouping_col], observed=True)

    # 同じ条件で1着になるの確率を計算
    bunsi1 = grouped1[target_col].cumsum() - grouped2[target_col].cumsum()
    bunbo1 = grouped1[target_col].cumcount() - grouped2[target_col].cumcount()

    df[feature_name] = bunsi1 / bunbo1.replace(0, np.nan) 
    feature_col.append(feature_name)

    return df, feature_col


# TrueSKillを計算する関数
def calc_trueskill_fast(df_to_copy, feature_col, target_col, prefix):
    """
    TrueSkill計算を効率的なDataFrame操作で高速化したバージョン。
    """
    if (target_col is None) or (prefix is None):
        raise ValueError("target_col and prefix must be specified")

    df = df_to_copy.copy()
    
    # 元のfeature_colリストを変更しないように新しいリストを作成
    new_feature_col = feature_col.copy()

    CONFIDENCE_MULTIPLIER = 3

    # 新しく追加する列名を事前に定義
    ts_mu_col = f"{prefix}_TrueSkill"
    ts_sigma_col = f"{prefix}_TrueSkill_sigma"
    ts_min_col = f"{prefix}_TrueSkill_min"
    ts_max_col = f"{prefix}_TrueSkill_max"
    ts_after_col = f"{prefix}_TrueSkill_after_racing"
    
    # 新しい特徴量をリストに追加
    new_feature_col.extend([ts_mu_col, ts_sigma_col, ts_min_col, ts_max_col])

    # TrueSkill環境とレーティング辞書を初期化
    env = TrueSkill(draw_probability=0.0)
    ratings = defaultdict(lambda: env.create_rating())

    # 処理済みのグループを格納するリスト
    processed_groups = []
    
    # groupbyオブジェクトを作成（sort=Falseで元の順序を維持）
    grouped = df.groupby("id_for_fold", observed=True, sort=False)

    for race_id, group in grouped:
        # 各グループのコピーに対して変更を加える
        group_copy = group.copy()
        
        all_targets = group_copy[target_col]

        # --- 1. レース前TrueSkillを効率的に記録 ---
        # .map()とlambda式を使い、辞書からmuとsigmaの値を高速に取得
        mu_series = all_targets.map(lambda x: ratings[x].mu)
        sigma_series = all_targets.map(lambda x: ratings[x].sigma)

        # 取得したSeriesをDataFrameの列として一括で代入
        group_copy[ts_mu_col] = mu_series
        group_copy[ts_sigma_col] = sigma_series
        
        # min/maxをベクトル演算で効率的に計算
        group_copy[ts_min_col] = mu_series - sigma_series * CONFIDENCE_MULTIPLIER
        group_copy[ts_max_col] = mu_series + sigma_series * CONFIDENCE_MULTIPLIER

        # --- 2. TrueSkill計算とレーティング更新 ---
        # 正常なレースデータのみを対象
        race_data = group_copy[group_copy["error_code"] == 0]
        
        # 意味のあるレーティング更新は、通常2つ以上のエンティティが存在する場合
        if len(race_data) >= 2:
            target_list = race_data[target_col].tolist()
            # ratings辞書から現在のレーティングオブジェクトのリストを作成
            race_ratings = [[ratings[target]] for target in target_list]
            ranks = race_data["rank"].tolist()

            # TrueSkillライブラリで新しいレーティングを計算
            new_ratings = env.rate(race_ratings, ranks=ranks)

            # ratings辞書を新しいレーティングで更新
            for target, new_rating_tuple in zip(target_list, new_ratings):
                ratings[target] = new_rating_tuple[0]
        
        # --- 3. レース後TrueSkillを効率的に記録 ---
        # 更新後のratings辞書からmuの値を.map()で取得
        group_copy[ts_after_col] = all_targets.map(lambda x: ratings[x].mu)

        # 処理済みのグループをリストに追加
        processed_groups.append(group_copy)

    # --- 4. 最後に処理済みグループを一度に結合 ---
    # .sort_values(by="datetime")で安全性を確保
    result_df = pd.concat(processed_groups).sort_values(by="datetime", ascending=True)

    return result_df, new_feature_col



# Elo Rating の計算用の関数
def calc_elo_rating_fast(df_to_copy, feature_col, K=32, target_col=None, prefix=None):
    """
    Eloレーティング計算をNumPyによるベクトル化と効率的なDataFrame操作で高速化したバージョン。
    """
    if (target_col is None) or (prefix is None):
        raise ValueError("target_col and prefix must be specified")

    df = df_to_copy.copy()
    
    # 元のfeature_colリストを変更しないように新しいリストを作成
    new_feature_col = feature_col.copy()
    new_feature_col.append(f"{prefix}_EloRating")

    # レート保存用辞書
    ratings = defaultdict(lambda: 1500)
    
    # 処理済みのグループを格納するリスト
    processed_groups = []
    
    # groupbyオブジェクトを一度だけ作成
    grouped = df.groupby("id_for_fold", observed=True, sort=False)
    
    for race_id, group in grouped:
        # groupをコピーして変更を加えることで、元のDataFrameへの意図しない変更を防ぐ
        group_copy = group.copy()

        # --- 1. レース前Eloレーティングを効率的に記録 ---
        # .map()は辞書を使った高速な値のマッピングを提供します
        group_copy[f"{prefix}_EloRating"] = group_copy[target_col].map(ratings)

        # Elo計算対象の正常なレースデータ
        race_data = group_copy[group_copy["error_code"] == 0]
        
        # 出走頭数が2頭未満の場合はEloの変動なし
        if len(race_data) < 2:
            group_copy[f"{prefix}_EloRating_after_racing"] = group_copy[f"{prefix}_EloRating"]
            processed_groups.append(group_copy)
            continue

        # --- 2. NumPyによるElo計算のベクトル化 ---
        # 必要なデータをNumPy配列として抽出
        horses = race_data[target_col].values
        ranks = race_data["rank"].values
        pre_ratings = race_data[f"{prefix}_EloRating"].values
        
        n_horses = len(horses)
        K_modified = K / (n_horses - 1)

        # 全ての馬のペアのインデックスを一度に生成
        # np.triu_indicesはcombinations(range(n_horses), 2)と等価なインデックスペアを高速に生成します
        idx_i, idx_j = np.triu_indices(n_horses, k=1)

        # ペアごとのレーティングと着順をベクトルとして取得
        R_i, R_j = pre_ratings[idx_i], pre_ratings[idx_j]
        rank_i, rank_j = ranks[idx_i], ranks[idx_j]

        # 期待勝率E_iをベクトル演算で一括計算
        E_i = 1 / (1 + 10 ** ((R_j - R_i) / 400))
        
        # 実際の勝敗S_iをベクトル演算で一括計算
        # np.whereを使って条件分岐を効率的に処理します
        # np.where(条件（配列同士で比較）, Trueの時の値, Falseの時の値) -> np.array
        S_i = np.where(rank_i < rank_j, 1.0, np.where(rank_i > rank_j, 0.0, 0.5))

        # 各ペアにおけるEloレーティングの変動値を一括計算
        delta_for_i = K_modified * (S_i - E_i)
        
        # 各馬の総変動値を計算
        # np.add.atは、同じインデックスに対して値を安全に加算できるため、
        # 各馬が複数のペアに含まれる場合の合計デルタを計算するのに適しています。
        delta_array = np.zeros(n_horses, dtype=np.float64)
        np.add.at(delta_array, idx_i, delta_for_i)
        np.add.at(delta_array, idx_j, -delta_for_i) # delta_j は -delta_i となります

        # --- 3. レーティングの一括更新 ---
        for i, horse in enumerate(horses):
            ratings[horse] += delta_array[i]

        # --- 4. レース後Eloレーティングを効率的に記録 ---
        group_copy[f"{prefix}_EloRating_after_racing"] = group_copy[target_col].map(ratings)
        
        processed_groups.append(group_copy)

    # --- 5. 最後に処理済みグループを一度に結合 ---
    # .sort_values(by="datetime")で安全性を確保
    result_df = pd.concat(processed_groups).sort_values(by="datetime", ascending=True)

    return result_df, new_feature_col



# glicko2を計算する関数(なぜかNanの列が存在するので要確認)
def calc_glicko2_common(
    df_to_copy, feature_col, target_col, prefix,
    *, conf_mult=3.0, rd_cap=350.0, rd_floor=30.0,
    init_mu=1500.0, init_rd=250.0, init_vol=0.06,
    rating_period_days=30
):
    df = df_to_copy.copy()
    feature_col = feature_col.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    main  = f"{prefix}_Glicko2"
    rd    = f"{prefix}_Glicko2_RD"
    gmin  = f"{prefix}_Glicko2_min"
    gmax  = f"{prefix}_Glicko2_max"
    after = f"{prefix}_Glicko2_after_racing"
    df[[main, rd, gmin, gmax, after]] = np.nan
    feature_col.extend([main, rd, gmin, gmax])

    players = defaultdict(lambda: Player(rating=init_mu, rd=init_rd, vol=init_vol))
    last_played = dict()

    for race_id, group in df.groupby("id_for_fold", observed=True):
        race_date = group["datetime"].iloc[0]
        horses_all = group[target_col].tolist()

        # 経過日数による RD 拡散
        for h in horses_all:
            if h in last_played:
                diff_days = (race_date - last_played[h]).days
                n_periods = diff_days // rating_period_days
                for _ in range(int(n_periods)):
                    players[h].did_not_compete()

        # レース前レーティングの埋め込み（全馬対象）
        mu_pre = np.array([players[h].getRating() for h in horses_all])
        rd_pre = np.array([np.clip(players[h].getRd(), rd_floor, rd_cap) for h in horses_all])
        idx = group.index
        df.loc[idx, main] = mu_pre
        df.loc[idx, rd]   = rd_pre
        df.loc[idx, gmin] = mu_pre - conf_mult * rd_pre
        df.loc[idx, gmax] = mu_pre + conf_mult * rd_pre

        # 正常な馬のみでレート更新
        race = group[group["error_code"] == 0]
        horses_valid, ranks = race[target_col].tolist(), race["rank"].tolist()
        if len(horses_valid) < 2:
            # 出走日だけ記録して終わり
            for h in horses_valid:
                last_played[h] = race_date
            continue

        # 勝敗作成
        update_args = {h: ([], [], []) for h in horses_valid}
        for (h_i, r_i), (h_j, r_j) in combinations(zip(horses_valid, ranks), 2):
            s_i, s_j = (1.0, 0.0) if r_i < r_j else (0.0, 1.0) if r_i > r_j else (0.5, 0.5)
            update_args[h_i][0].append(players[h_j].getRating())
            update_args[h_i][1].append(np.clip(players[h_j].getRd(), rd_floor, rd_cap))
            update_args[h_i][2].append(s_i)
            update_args[h_j][0].append(players[h_i].getRating())
            update_args[h_j][1].append(np.clip(players[h_i].getRd(), rd_floor, rd_cap))
            update_args[h_j][2].append(s_j)

        # 一括更新
        for h, (opp_r, opp_rd, score) in update_args.items():
            try:
                players[h].update_player(opp_r, opp_rd, score)
            except ZeroDivisionError:
                pass
            last_played[h] = race_date

        # レース後レーティングの埋め込み（全馬対象）
        mu_after = np.array([players[h].getRating() for h in horses_all])
        df.loc[idx, after] = mu_after

    return df, feature_col


# レーティングの上昇量を計算
def calc_rating_diff(df, feature_col, target_col=None, prefix=None):
    if (target_col is None) or (prefix is None):
        raise ValueError("target_col and prefix must be selected")
    
    df = df.copy()
    feature_col = feature_col.copy()

    last_trueskill = df.groupby("horse", observed=True)[target_col]
    for i in [1, 3]:
        feature_name = f"{prefix}_diff_from_last{i}_racing"
        df[feature_name] = df[target_col] - last_trueskill.shift(i)
        feature_col.append(feature_name)

    return df, feature_col


# オッズデータと結合する関数
def merge_last_N_odds(df, feature_col, tansho_odds_path=None):
    if tansho_odds_path is None:
        raise ValueError("Error in merge_last_N_odds: tansho_odds_path must be specified")
    # 過学習抑制の為にオッズ情報をなるべく加えない（5分前オッズのみを加える）。
    odds_df = pd.read_csv(tansho_odds_path, encoding="shift-jis")
    odds_df_selected =  odds_df[["race_id", "tansho_odds_20"]]

    new_feature_col = feature_col + ["tansho_odds_20"]

    df = pd.merge(left=df, right=odds_df_selected, how="left", on=["race_id"])

    return df, new_feature_col