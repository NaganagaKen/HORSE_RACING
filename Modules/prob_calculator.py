import pandas as pd
from itertools import combinations_with_replacement

# 枠連の期待値を計算しやすいようにする関数
def calc_waku_pred(df_pred): # df_predが与えられる想定
    #　df_pred はyear, month, day, horse, 
    # pred_class_1, pred_class_2, pred_class_3,
    # target, race_id, waku_num をデータとして含む必要がある。
    df_pred = df_pred.copy()
    
    df_pred["id_for_fold"] = df_pred["race_id"] // 100
    
    odds_df = pd.read_csv("../Data/wakuren/wakuren_2021_2025.csv")
    odds_df = odds_df[["waku_pair", "wakuren_confirmed_odds", "id_for_fold"]]

    calced_df = []

    # 1レースずつ計算
    for id in df_pred["id_for_fold"].unique():
        race_data = df_pred[df_pred.id_for_fold == id]
        
        # 各枠連の的中率を計算
        race_prob = dict()
        for i, j in combinations_with_replacement(list(range(1,9)), 2):
            prob_sum = 0
            waku_i = race_data[race_data["waku_num"] == i]
            waku_j = race_data[race_data["waku_num"] == j]

            if i != j: # ゾロ目じゃない場合は普通に全通り計算
                for waku_i_idx in range(len(waku_i)):
                    for waku_j_idx in range(len(waku_j)):
                        horse1 = waku_i.iloc[waku_i_idx, :]
                        horse2 = waku_j.iloc[waku_j_idx, :]
                        prob_sum += horse1["pred_class_1"] * (horse2["pred_class_2"] / (1-horse1["pred_class_2"]))
                        prob_sum += horse2["pred_class_1"] * (horse1["pred_class_2"] / (1-horse2["pred_class_2"]))

            elif i == j and len(waku_i) >= 2: #ゾロ目の場合は別で計算(2頭以上いないと計算できない)
                horse1 = waku_i.iloc[0, :]
                horse2 = waku_i.iloc[1, :]
                prob_sum += horse1["pred_class_1"] * (horse2["pred_class_2"] / (1-horse1["pred_class_2"]))
                prob_sum += horse2["pred_class_1"] * (horse1["pred_class_2"] / (1-horse2["pred_class_2"]))
                

            race_prob[f"枠{i}-{j}"] = prob_sum

        # 1レース分のDataFrameを作成
        race_prob_df = pd.Series(race_prob).reset_index(drop=False).rename(columns={"index":"waku_pair", 0:"pred"})
        race_prob_df["id_for_fold"] = id
        
        # target情報の付加(これがないと期待値を計算できない)
        # 同着の場合は無視する(targetは全て0とする)
        try:
            win_waku1 = race_data[race_data["target"] == 1]["waku_num"].values[0]
            win_waku2 = race_data[race_data["target"] == 2]["waku_num"].values[0]
            win_waku_pair = f"枠{min(win_waku1, win_waku2)}-{max(win_waku2, win_waku1)}"
            race_prob_df["target"] = 0
            race_prob_df["target"] = race_prob_df["waku_pair"].apply(lambda x: 1 if x == win_waku_pair else 0)
        except:
            race_prob_df["target"] = 0
            print(id) # 同着の場合は無視

        calced_df.append(race_prob_df)

    concat_df = pd.concat(calced_df, axis=0) # 各レースを一つのdfにまとめる
    merged_df = pd.merge(left=concat_df, right=odds_df, how="left", on=["id_for_fold", "waku_pair"]) # オッズと結合

    return merged_df