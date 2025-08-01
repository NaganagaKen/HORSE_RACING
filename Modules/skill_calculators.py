import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from trueskill import TrueSkill
from myglicko2 import Player
from collections import defaultdict
from itertools import combinations


# レーティングシステムに対する処理を一括で行う関数
def all_rating_calculator(df, feature_col, ranking_col):
    df = df.copy()
    feature_col = feature_col.copy()
    ranking_col = ranking_col.copy()

    # TrueSkillの計算
    print("calculating horse trueskill is in progress")
    horse_ts_calculator = trueskill_calculator("horse", "horse")
    df, feature_col = horse_ts_calculator.fit_transform(df, feature_col) ###
    print("calculating jockey trueskill is in progress")
    jockey_ts_calculator = trueskill_calculator("jockey_id", "jockey")
    df, feature_col = jockey_ts_calculator.fit_transform(df, feature_col) ###

    # EloRatingの計算
    print("calculating horse EloRating is in progress")
    horse_er_calculator = elorating_calculator(target_col="horse", prefix="horse")
    df, feature_col = horse_er_calculator.fit_transform(df, feature_col)
    print("calculating jockey EloRating is in progress")
    jockey_er_calculator = elorating_calculator(target_col="jockey_id", prefix="jockey")
    df, feature_col = jockey_er_calculator.fit_transform(df, feature_col)

    # Glicko2の計算
    print("calculating horse Glicko2 is in progress")
    horse_g2_calculator = glicko2_calculator(target_col="horse", prefix="horse", rating_period_days=30)
    df, feature_col = horse_g2_calculator.fit_transform(df, feature_col)
    # jockey Glicko2は不安定なので、要改善
    print("calculating jockey Glicko2 is in progress")
    jockey_g2_calculator = glicko2_calculator(target_col="jockey_id", prefix="jockey", rating_period_days=14,
                                         rd_cap=100, rd_floor=20, conf_mult=3.0)
    df, feature_col = jockey_g2_calculator.fit_transform(df, feature_col)

    # 各レートの上昇量を計算
    rating_diff_list = ['horse_TrueSkill', 'jockey_TrueSkill', 'horse_EloRating', 'jockey_EloRating', 'horse_Glicko2'] # 空白区切り
    for col in rating_diff_list:
        df, feature_col = calc_rating_diff(df, feature_col, target_col=col, prefix=col)


    # レーティングの相互作用特徴量を追加
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_list = ['horse_TrueSkill', 'horse_TrueSkill_min', 'horse_TrueSkill_max',  # horseのTrueSkill
                 'jockey_TrueSkill', 'jockey_TrueSkill_min', 'jockey_TrueSkill_max', # jockeyのTrueSkill
                 'horse_EloRating', 'jockey_EloRating',  # EloRating
                 'horse_Glicko2', 'horse_Glicko2_min', 'horse_Glicko2_max', # horseのGlicko2
                 "jockey_Glicko2", "jockey_Glicko2_min", "jockey_Glicko2_max"] # jockeyのGlicko2
    poly_features = poly.fit_transform(df[poly_list])
    poly_features_name = poly.get_feature_names_out(poly_list)
    poly_features_df = pd.DataFrame(poly_features, columns=poly_features_name, index=df.index)

    # 15 個の元列だけ除外してから結合（PolynomialFeaturesは相互作用を計算しないものまで含めてしまう）
    interaction_cols = [c for c in poly_features_name if c not in poly_list]
    feature_col.extend(interaction_cols)
    ranking_col.extend(["jockey_TrueSkill horse_Glicko2", "jockey_TrueSkill_min horse_Glicko2", "jockey_TrueSkill_max horse_Glicko2"])
    df = pd.concat([df, poly_features_df[interaction_cols]], axis=1)

    print("poly calculated")

    return df, feature_col, ranking_col


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


# trueskillを計算するクラス
class trueskill_calculator:
    def __init__(self, target_col, prefix, CONFIDENCE_MULTIPLIER=3):
        self.target_col = target_col # dataframe内のどの列のtrueskillを計算するか
        self.prefix = prefix # 列名の指定
        self.CONFIDENCE_MULTIPLIER = CONFIDENCE_MULTIPLIER # シグマ範囲
    

    def fit_transform(self, df, feature_col):
        df = df.copy()
        new_feature_col = feature_col.copy()

        target_col = self.target_col
        prefix = self.prefix
        CONFIDENCE_MULTIPLIER = self.CONFIDENCE_MULTIPLIER

        # 新しく追加する列名を事前に定義
        ts_mu_col = f"{prefix}_TrueSkill"
        ts_sigma_col = f"{prefix}_TrueSkill_sigma"
        ts_min_col = f"{prefix}_TrueSkill_min"
        ts_max_col = f"{prefix}_TrueSkill_max"
        ts_after_col = f"{prefix}_TrueSkill_after_racing" # これはDataFrameには含めるけど、特徴量としては含まれない。
        
        # 新しい特徴量をリストに追加
        new_feature_col.extend([ts_mu_col, ts_sigma_col, ts_min_col, ts_max_col])

        # TrueSkill環境とレーティング辞書を初期化
        # 後で呼び出せるようにしておく
        self.env = TrueSkill(draw_probability=0.0)
        env = self.env
        self.ratings = defaultdict(lambda: env.create_rating())
        ratings = self.ratings

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

    # 新しいレースで予測したい場合はこちらを呼び出す
    def transform(self, df, feature_col):
        df = df.copy()
        new_feature_col = feature_col.copy()

        env = self.env
        ratings = self.ratings
        prefix = self.prefix
        target_col = self.target_col
        CONFIDENCE_MULTIPLIER  = self.CONFIDENCE_MULTIPLIER

        # ターゲット名を指定
        ts_mu_col = f"{prefix}_TrueSkill"
        ts_sigma_col = f"{prefix}_TrueSkill_sigma"
        ts_min_col = f"{prefix}_TrueSkill_min"
        ts_max_col = f"{prefix}_TrueSkill_max"
        new_feature_col.extend([ts_mu_col, ts_sigma_col, ts_min_col, ts_max_col])


        all_targets = df[target_col]

        # --- 1. レース前TrueSkillを効率的に記録 ---
        # .map()とlambda式を使い、辞書からmuとsigmaの値を高速に取得
        mu_series = all_targets.map(lambda x: ratings[x].mu)
        sigma_series = all_targets.map(lambda x: ratings[x].sigma)

        # 取得したSeriesをDataFrameの列として一括で代入
        df[ts_mu_col] = mu_series
        df[ts_sigma_col] = sigma_series
        
        # min/maxをベクトル演算で効率的に計算
        df[ts_min_col] = mu_series - sigma_series * CONFIDENCE_MULTIPLIER
        df[ts_max_col] = mu_series + sigma_series * CONFIDENCE_MULTIPLIER

        return df, new_feature_col
    


# Elo Rating計算するクラス
class elorating_calculator:
    def __init__(self, K=32, target_col=None, prefix=None):
        if (target_col is None) or (prefix is None):
            raise ValueError("target_col and prefix must be specified")
        self.K = K
        self.target_col = target_col
        self.prefix = prefix
    
    def fit_transform(self, df, feature_col):
        K = self.K
        target_col = self.target_col
        prefix = self.prefix
        
        # 元のデータを変更しないようにコピーを作成しておく
        df = df.copy()
        new_feature_col = feature_col.copy()
        new_feature_col.append(f"{prefix}_EloRating")

        # レート保存用辞書
        self.ratings = defaultdict(lambda: 1500)
        ratings = self.ratings
        
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
    

    # 出走前レース用の関数
    def transform(self, df, feature_col):
        df = df.copy()
        feature_col = feature_col.copy()

        ratings = self.ratings
        target_col = self.target_col
        prefix = self.prefix

        df[f"{prefix}_EloRating"] = df[target_col].map(ratings)
        feature_col.append(f"{prefix}_EloRating")

        return df, feature_col
    


# Glicko2を計算する関数
class glicko2_calculator:
    def __init__(self, target_col=None, prefix=None, rating_period_days=30,
                 rd_cap=350.0, rd_floor=40.0, conf_mult=3.0):
        if (target_col is None) or (prefix is None):
            raise ValueError("target_col and prefix must be specified")
        
        self.target_col = target_col
        self.prefix = prefix
        self.rating_period_days = rating_period_days
        self.rd_cap = rd_cap
        self.rd_floor = rd_floor
        self.conf_mult = conf_mult
    
    def fit_transform(self, df, feature_col):
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        df["date"] = df["datetime"].dt.floor("D")   # ← 1 日粒度キーを追加
        feature_col = feature_col.copy()

        # ---------- 可視化パラメータ ----------
        rating_floor = 1000.0       # 表示だけ下限 1000 に丸める
        conf_mult = self.conf_mult
        rd_cap    = self.rd_cap
        rd_floor  = self.rd_floor
        # --------------------------------------

        main   = f"{self.prefix}_Glicko2"
        viz    = f"{self.prefix}_Glicko2_viz"
        rd_col = f"{self.prefix}_Glicko2_RD"
        gmin   = f"{self.prefix}_Glicko2_min"
        gmax   = f"{self.prefix}_Glicko2_max"
        after  = f"{self.prefix}_Glicko2_after_racing"
        df[[main, viz, rd_col, gmin, gmax, after]] = np.nan
        feature_col.extend([main, rd_col, gmin, gmax])  # viz 列は学習に使わない

        # ---- プレイヤー初期化 ----
        players = defaultdict(lambda: Player(rating=1500.0, rd=250.0, vol=0.06))
        last_played = {}

        # ===== 1 日まとめて更新 =====
        for day, day_grp in df.groupby("date", observed=True):
            # ---------- RD 拡散 ----------
            for h in day_grp[self.target_col].unique():
                if h in last_played:
                    diff_days = (day - last_played[h]).days
                    n_periods = min(diff_days // self.rating_period_days, 6)
                    for _ in range(int(n_periods)):
                        players[h].did_not_compete()

            # ---------- レース前の埋め込み ----------
            mu_raw = np.array([players[h].getRating() for h in day_grp[self.target_col]])
            rd_pre = np.array([np.clip(players[h].getRd(), rd_floor, rd_cap)
                               for h in day_grp[self.target_col]])
            idx = day_grp.index
            df.loc[idx, main] = mu_raw                 # 学習用（raw 値）
            df.loc[idx, viz]  = np.clip(mu_raw, rating_floor, None)  # 可視化用
            df.loc[idx, rd_col] = rd_pre
            df.loc[idx, gmin] = mu_raw - conf_mult * rd_pre
            df.loc[idx, gmax] = mu_raw + conf_mult * rd_pre

            # ---------- 1 日分の対戦ログを蓄積 ----------
            daily_args = {h: ([], [], []) for h in day_grp[self.target_col].unique()}

            for _, race in day_grp.groupby("id_for_fold", observed=True):
                race = race[race["error_code"] == 0]
                horses, ranks = race[self.target_col].tolist(), race["rank"].tolist()
                if len(horses) < 2:
                    continue

                for (h_i, r_i), (h_j, r_j) in combinations(zip(horses, ranks), 2):
                    s_i, s_j = (1.0, 0.0) if r_i < r_j else (0.0, 1.0) if r_i > r_j else (0.5, 0.5)
                    daily_args[h_i][0].append(players[h_j].getRating())
                    daily_args[h_i][1].append(np.clip(players[h_j].getRd(), rd_floor, rd_cap))
                    daily_args[h_i][2].append(s_i)
                    daily_args[h_j][0].append(players[h_i].getRating())
                    daily_args[h_j][1].append(np.clip(players[h_i].getRd(), rd_floor, rd_cap))
                    daily_args[h_j][2].append(s_j)

            # ---------- 1 日 1 回だけの更新 ----------
            for h, (opp_r, opp_rd, score) in daily_args.items():
                if score:                       # 出走があれば更新
                    players[h].update_player(opp_r, opp_rd, score)
                    last_played[h] = day

            # ---------- レース後の可視化値 ----------
            mu_after = np.array([players[h].getRating() for h in day_grp[self.target_col]])
            df.loc[idx, after] = mu_after

        return df, feature_col

    

    # 新しいレース用のレーディング埋め込み関数(1レースずつ)
    def transform(self, df, feature_col):
        df = df.copy()
        feature_col = feature_col.copy()
        
        target_col = self.target_col
        prefix = self.prefix

        # パラメータの設定
        conf_mult=3.0
        rd_cap=350.0
        rd_floor=30.0
        rating_period_days=30

        main  = f"{prefix}_Glicko2"
        rd    = f"{prefix}_Glicko2_RD"
        gmin  = f"{prefix}_Glicko2_min"
        gmax  = f"{prefix}_Glicko2_max"

        players = self.ratings
        last_played = self.last_played
        race_date = df["datetime"].iloc[0]
        horses_all = df[target_col].tolist()

        # RDの拡散
        # 経過日数による RD 拡散
        for h in horses_all:
            if h in last_played:
                diff_days = (race_date - last_played[h]).days
                n_periods = diff_days // rating_period_days
                for _ in range(int(n_periods)):
                    players[h].did_not_compete()


        mu_pre = np.array([players[h].getRating() for h in horses_all])
        rd_pre = np.array([np.clip(players[h].getRd(), rd_floor, rd_cap) for h in horses_all])
        idx = df.index
        df.loc[idx, main] = mu_pre
        df.loc[idx, rd]   = rd_pre
        df.loc[idx, gmin] = mu_pre - conf_mult * rd_pre
        df.loc[idx, gmax] = mu_pre + conf_mult * rd_pre

        feature_col.extend([main, rd, gmin, gmax])

        return df, feature_col