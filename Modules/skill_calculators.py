import numpy as np
import pandas as pd
from trueskill import TrueSkill
from collections import defaultdict


class trueskill_calcluator:
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