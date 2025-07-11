import pandas as pd
from itertools import combinations_with_replacement

# 枠連の期待値を計算しやすいようにする関数
def calc_waku_pred(df): # df_predが与えられる想定
    df = df.copy()
    
    df["id_for_fold"] = df["race_id"] // 100
    
    odds_df = pd.read_csv("../Data/wakuren/wakuren_2025.csv")
    odds_df = odds_df[["waku_pair", "wakuren_confirmed_odds", "id_for_fold"]]

    calced_df = []

    for id in df["id_for_fold"].unique():
        group = df[df.id_for_fold == id]
        odds = odds_df[odds_df.id_for_fold == id]

        sum_in_waku = group.groupby("waku_num", observed=True) \
            [["pred_class_1", "pred_class_2"]] \
            .sum().reset_index(drop=False)
        
        sum_dict = dict()
        for i, j in list(combinations_with_replacement(sum_in_waku["waku_num"].unique().tolist(), 2)):
            sum_of_prob = 0
            if i != j:
                sum_of_prob += sum_in_waku[sum_in_waku.waku_num == i]["pred_class_1"].values[0] \
                    * sum_in_waku[sum_in_waku.waku_num == j]["pred_class_2"].values[0]
                sum_of_prob += sum_in_waku[sum_in_waku.waku_num == j]["pred_class_1"].values[0] \
                    * sum_in_waku[sum_in_waku.waku_num == i]["pred_class_2"].values[0]
            else:
                sum_of_prob = sum_in_waku[sum_in_waku.waku_num == i]["pred_class_1"].values[0] \
                    * sum_in_waku[sum_in_waku.waku_num == i]["pred_class_2"].values[0]
                
            sum_dict[f"枠{i}-{j}"] = sum_of_prob

        waku_pred = pd.Series(sum_dict).reset_index(drop=False).rename(columns={"index":"waku_pair", 0:"pred"})
        odds_merged = pd.merge(left=waku_pred, right=odds, how="inner", on=["waku_pair"])

        target_set = set(group[(group.target==1) | (group.target==2)]["waku_num"].values.tolist())
        if len(target_set) == 1:
            target1 = target_set.pop()
            target2 = target1
        elif len(target_set) == 2:
            target1, target2 = target_set

        odds_merged["target"] = 0
        odds_merged["target"] = odds_merged["waku_pair"] \
            .apply(lambda x: 1 if x == f"枠{min(target1,target2)}-{max(target1,target2)}" else 0)
        
        calced_df.append(odds_merged)

    ret_df = pd.concat(calced_df, axis=0)

    return ret_df
