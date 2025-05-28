import pandas as pd
import numpy as np
from my_modules import GroupTimeSeriesSplit
import optuna
from sklearn.metrics import log_loss
import lightgbm as lgb
from lightgbm import LGBMClassifier


def simple_lightGBM(df, feature_col):
    splitter = GroupTimeSeriesSplit(n_splits=5)

    # 確定パラメータの定義
    params = {
        "objective" : "binary",
        "metric" : "binary_logloss",
        "n_estimators": 1000,
        "n_jobs" : -1,
        "verbose" : -1,
        }

    # データの前処理
    cat_col = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_col:
        df[col] = df[col].astype("category")

    logloss_of_each_horse_N = []
    models = []

    # ---- ここから各horse_Nに対するループ ----

    # 各horse_Nごとにモデルを作成
    for i in sorted(df.horse_N.unique().tolist()) :
      print(f"Start tuning of horse_{i}")

      df_horse_N = df[df.horse_N == i].copy()

      # 訓練データ（パラメータチューニング用）とテストデータの分割
      X, y = df_horse_N.drop(["target"], axis=1), df_horse_N["target"]
      X_train, X_test, y_train, y_test = train_test_group_split(X, y, test_size=0.3)

      # パラメータチューニング開始
      objective = create_objective(X_train, y_train, splitter=splitter, feature_col=feature_col, params=params)
      study = optuna.create_study(
          direction="minimize",
          sampler=optuna.samplers.TPESampler() # パラメータ探索範囲を確定するには、ランダムサンプラーにする。
          )
      study.optimize(objective, n_trials=100) # 最適化の実行

      # 調整後のパラメータの表示
      print("Best params : ", study.best_params)

      # 可視化
      optuna.visualization.plot_param_importances(study).show() # feature_importanceみたいなやつ
      optuna.visualization.plot_slice(
          study,
          params=["max_bin", "num_leaves", "min_data_in_leaf", "min_sum_hessian_in_leaf",
                  "bagging_fraction", "bagging_freq", "feature_fraction", "lambda_l1", "lambda_l2",
                  "min_gain_to_split", "max_depth", "learning_rate", "path_smooth"]
      ).show()

      # 全体を学習させる
      kwargs = {**params, **study.best_params}
      model = LGBMClassifier(**kwargs)
      model.fit(X[feature_col], y)
      pred = model.predict_proba(X_test[feature_col])[:, 1]
      models.append(tuple([i, model])) # 番号（horse_N）とmodelを格納

      # テストデータでloglossを計算
      prob_calcurate = prob_calculator(X_test, pred)
      logloss = log_loss(y_test, prob_calcurate["first_prize_prob"])
      logloss_of_each_horse_N.append(tuple([i, logloss]))

      print(f"End tuning of horse:{i}")

    # ---- ループ終了 ----

    # 各horse_Nのloglossを表示
    for i, logloss_score in logloss_of_each_horse_N:
      print(f"logloss of horse_{i} : ", logloss_score)

    return models


# objective関数を作る関数
def create_objective(X, y, splitter, feature_col, params):
    def objective(trial):
        # パラメータの定義
        tuning_params = {
            "max_bin": trial.suggest_int("max_bin", 10, 255),
            "num_leaves": trial.suggest_int("num_leaves", 2, 100),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-8, 10),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
            "max_depth": trial.suggest_int("max_depth", 2, 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
            "path_smooth": trial.suggest_float("path_smooth", 0, 10)
        }
        kwargs = {**params, **tuning_params}

        # 交差検証
        oof_preds = np.zeros(len(X))
        oof_preds[:] = np.nan
        for tr_idx, val_idx in splitter.split(X, y, groups=X["id_for_fold"]):

            X_train, y_train = X.iloc[tr_idx, :], y.iloc[tr_idx]
            X_test, y_test = X.iloc[val_idx, :], y.iloc[val_idx]

            # 要らない列を削除
            X_train, X_test = X_train[feature_col], X_test[feature_col]

            # modelの宣誓とコールバックの定義
            model = LGBMClassifier(**kwargs)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)] # 50ラウンドで停止

            # modelの学習
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)], # 評価データセットを指定
                      eval_metric="binary_logloss",
                      callbacks=callbacks)

            # 予測値の取得
            oof_preds[val_idx] = model.predict_proba(X_test)[:, 1]

        # NaN ではない部分のインデックスを取得
        not_nan_indices = ~np.isnan(oof_preds)

        # NaN ではない部分のみで log_loss を計算
        # y も対応するインデックスでフィルタリングする
        oof_df_for_prob_calc = X.loc[not_nan_indices].copy()
        oof_preds_normalized_df = prob_calculator(oof_df_for_prob_calc, oof_preds[not_nan_indices])
        avg_logloss = log_loss(y[not_nan_indices], oof_preds_normalized_df["first_prize_prob"])

        return avg_logloss

    return objective


# 予測確率を正規化する関数
def prob_calculator(df, prob):
    df_copy = df.copy()

    # 予測確率をクリッピングして、完全に0や1にならないようにする (LogLossの安定化)
    epsilon = 1e-15 # ごく小さい値
    df_copy["unnormalized_prob"] = np.clip(prob, epsilon, 1 - epsilon)
    # FutureWarningを避けるためにobserved=Falseを追加
    sum_prob = df_copy.groupby(["id_for_fold"], observed=False)["unnormalized_prob"].sum()
    sum_prob = pd.DataFrame({"sum_prob": sum_prob})
    df_copy = pd.merge(df_copy, sum_prob, how="left", on="id_for_fold")
    df_copy["first_prize_prob"] = df_copy["unnormalized_prob"] / df_copy["sum_prob"]

    df_copy = df_copy.drop(["unnormalized_prob", "sum_prob"], axis=1)
    return df_copy


# グループごとの訓練・テストデータの分割
def train_test_group_split(X, y, test_size=0.2, groups="id_for_fold"):

    unique_group = X[groups].unique()
    n_groups = X[groups].nunique()
    n_train = round(n_groups * (1-test_size))
    s_groups = pd.Series(X[groups])

    train_groups = unique_group[:n_train]
    test_groups = unique_group[n_train:]

    train_idx = np.where(s_groups.isin(train_groups))[0]
    test_idx = np.where(s_groups.isin(test_groups))[0]

    X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test