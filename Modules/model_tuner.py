import pandas as pd
import numpy as np
from my_modules import GroupTimeSeriesSplit
import matplotlib.pyplot as plt
from IPython.display import display
import optuna
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import lightgbm as lgb
from lightgbm import LGBMClassifier
import time
import random

plt.rcParams['font.family'] = 'Yu Gothic'


def simple_lightGBM(df, feature_col, visualization=False, memo="None", scores_path="../Memo/logloss_score_of_each_horse_N.csv",
                    n_trials=100, save_result=True):
    splitter = GroupTimeSeriesSplit(n_splits=5)
    
    # 乱数シードを固定
    np.random.seed(42)
    random.seed(42)


    # 確定パラメータの定義
    params = {
        "objective" : "binary",
        "metric" : "binary_logloss",
        "n_estimators": 5000,
        "n_jobs" : -1,
        "verbose" : -1,
        "random_state" : 42
        }

    # データの前処理
    cat_col = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_col:
        df[col] = df[col].astype("category")

    logloss_of_each_horse_N = []

    # ---- ここから学習開始 ----

    # 訓練データ（パラメータチューニング用）とテストデータの分割
    X, y = df.drop(["target"], axis=1), df["target"]
    X_train, X_test_cal, y_train, y_test_cal = train_test_group_split(X, y, test_size=0.4)
    X_cal, X_test, y_cal, y_test = train_test_group_split(X_test_cal, y_test_cal, test_size=0.3)

    # パラメータチューニング開始
    objective = create_objective(X_train, y_train, splitter=splitter, feature_col=feature_col, params=params)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42) # パラメータ探索範囲を確定するには、ここをランダムサンプラーにする。
        )
    study.optimize(objective, n_trials=n_trials) # 最適化の実行

    # 調整後のパラメータの表示
    print("Best params : ", study.best_params)

    # 可視化
    if (visualization) :
        optuna.visualization.plot_param_importances(study).show() # feature_importanceみたいなやつ
        optuna.visualization.plot_slice(
            study,
            params=["max_bin", "num_leaves", "min_data_in_leaf", "min_sum_hessian_in_leaf",
                    "bagging_fraction", "bagging_freq", "feature_fraction", "lambda_l1", "lambda_l2",
                    "min_gain_to_split", "max_depth", "learning_rate", "path_smooth"]
        ).show()


    # 訓練データ全体を学習 -> キャリブレーション -> 勝率を予測
    # 最適化したパラメータで全体を学習
    kwargs = {**params, **study.best_params}
    model = LGBMClassifier(**kwargs)
    model.fit(X_train[feature_col], y_train)
    # キャリブレーション
    frozen_model = FrozenEstimator(model)
    calibrator = CalibratedClassifierCV(frozen_model, method="isotonic", n_jobs=-1)
    calibrator.fit(X_cal[feature_col], y_cal)
    # 勝率を予測
    pred = calibrator.predict_proba(X_test[feature_col])[:, 1]


    # モデルの重要度を表示
    importances = model.booster_.feature_importance(importance_type="gain")
    feature_name = pd.Series(model.feature_name_)
    indices = np.argsort(importances)[::-1] 

    plt.figure(figsize=(40,6))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), feature_name[indices], rotation=90, fontsize="xx-small")

    plt.show()


    # テストデータで各horse_Nごとのloglossを計算
    prob_calcurated_df = prob_calculator(X_test, pred)
    prob_calcurated_df = prob_calcurated_df.set_index(X_test.index) 
    for i in sorted(X.horse_N.unique().tolist()):
        horse_N_index = X_test.horse_N == i
        prob_calcurate_horse_N = prob_calcurated_df.loc[horse_N_index, "pred"]
        logloss_score = log_loss(y_test[horse_N_index], prob_calcurate_horse_N)
        logloss_of_each_horse_N.append(tuple([i, logloss_score]))


    # データを書き込み
    if save_result:
        old_scores = pd.read_csv(scores_path)
        # 登録時間
        now = time.ctime()
        cnvtime = time.strptime(now)
        current_score = [time.strftime("%Y/%m/%d %H:%M", cnvtime), memo] # これにスコアを追加していく
        # loglossスコア
        logloss_list = [score for i, score in logloss_of_each_horse_N]
        current_score.extend(logloss_list)
        current_score.append(sum(logloss_list))
        # auc
        auc_score = roc_auc_score(y_test, pred)
        current_score.append(auc_score)

        # DataFrameに変換
        current_data_dict = dict()
        data_col_name = old_scores.columns.tolist()
        for data ,col_name in zip(current_score, data_col_name):
            current_data_dict[col_name] = data
        current_score_df = pd.DataFrame([current_data_dict])
        update_scores = pd.concat([current_score_df, old_scores], ignore_index=True)
        update_scores.to_csv("../Memo/logloss_score_of_each_horse_N.csv", index=False)
        display(update_scores.head(5))

    # 返すデータの設定（予測値を埋め込む）
    X_test = prob_calculator(X_test, pred)
    X_test.loc[:, "target"] = y_test.values

    return model, X_test


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
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "path_smooth": trial.suggest_float("path_smooth", 0, 10)
        }
        kwargs = {**params, **tuning_params}

        # 交差検証
        oof_preds = np.zeros(len(X))
        oof_preds[:] = np.nan
        for tr_idx, val_idx in splitter.split(X, y, groups=X["id_for_fold"]):

            X_train, y_train = X.iloc[tr_idx, :], y.iloc[tr_idx]
            X_test, y_test = X.iloc[val_idx, :], y.iloc[val_idx]

            # weightを指定
            sample_weight_train = X_train["sample_weight"].values
            sample_weight_val = X_test["sample_weight"].values

            # 要らない列を削除
            X_train, X_test = X_train[feature_col], X_test[feature_col]

            # modelの宣誓とコールバックの定義
            model = LGBMClassifier(**kwargs)
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)] # 50ラウンドで停止

            # modelの学習
            model.fit(X_train, y_train,
                      sample_weight=sample_weight_train,
                      eval_set=[(X_test, y_test)],
                      eval_metric="binary_logloss",
                      eval_sample_weight=[sample_weight_val],
                      callbacks=callbacks)

            # 予測値の取得
            oof_preds[val_idx] = model.predict_proba(X_test)[:, 1]

        # NaN ではない部分のインデックスを取得
        not_nan_indices = ~np.isnan(oof_preds)

        # NaN ではない部分のみで log_loss を計算
        # y も対応するインデックスでフィルタリングする
        oof_df_for_prob_calc = X.copy()
        oof_preds_normalized_df = prob_calculator(oof_df_for_prob_calc[not_nan_indices], oof_preds[not_nan_indices])
        normalized_prob = oof_preds_normalized_df["pred"]
        log_loss_weight = X.loc[not_nan_indices, "sample_weight"]
        avg_logloss = log_loss(y[not_nan_indices], normalized_prob, sample_weight=log_loss_weight)

        return avg_logloss

    return objective


# 予測確率を正規化する関数
def prob_calculator(df_to_copy, prob):
    df = df_to_copy.copy()

    # 予測確率をクリッピングして、完全に0や1にならないようにする (LogLossの安定化)
    epsilon = 1e-15 # ごく小さい値
    df["unnormalized_prob"] = np.clip(prob, epsilon, 1 - epsilon)
    # FutureWarningを避けるためにobserved=Falseを追加
    sum_prob = df.groupby(["id_for_fold"], observed=False)["unnormalized_prob"].sum()
    sum_prob = pd.DataFrame({"sum_prob": sum_prob})
    df = pd.merge(df, sum_prob, how="left", on="id_for_fold")
    df.index = df_to_copy.index # mergeでインデックスがリセットされたので、元に戻す
    df["pred"] = df["unnormalized_prob"] / df["sum_prob"]

    df = df.drop(["unnormalized_prob", "sum_prob"], axis=1)
    return df


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