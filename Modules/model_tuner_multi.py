import pandas as pd
import numpy as np
import scipy.special
from my_modules import GroupTimeSeriesSplit
import matplotlib.pyplot as plt
from IPython.display import display
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb
from lightgbm import LGBMClassifier
import scipy
from scipy.optimize import minimize
import time
import random
import warnings

plt.rcParams['font.family'] = 'Yu Gothic'
# optunaのLightGBMPruningCallbackの警告を無視する設定
warnings.filterwarnings(
    "ignore",
    message=r"The reported value is ignored because this `step` .* is already reported\.",
    category=UserWarning,
    module="optuna"
)


# 多クラス分類用の関数
def multi_lightGBM(df, feature_col, visualization=False, memo="None", scores_path="../Memo/logloss_score_of_each_horse_N_multi.csv",
                    n_trials=100, save_result=True):
    
    splitter = GroupTimeSeriesSplit(n_splits=8)
    
    # 乱数シードを固定
    np.random.seed(42)
    random.seed(42)


    # 確定パラメータの定義
    params = {
        "objective" : "multiclass",
        "metric" : "multi_logloss",
        "n_estimators": 5000,
        "n_jobs" : -1,
        "verbose" : -1,
        "random_state" : 42
        }

    # データの前処理
    cat_col = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_col:
        df[col] = df[col].astype("category")

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
    study.optimize(objective, 
                   n_trials=n_trials) # 最適化の実行

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
    best_lgbm_params = study.best_params
    model = LGBMClassifier(**params, **best_lgbm_params)
    # 全体を学習
    model.fit(X_train[feature_col], y_train, 
              eval_set=[(X_cal[feature_col], y_cal)],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]) 

    # 過学習していないかチェック
    y_pred_train = model.predict_proba(X_train[feature_col])
    y_pred_cal = model.predict_proba(X_cal[feature_col])
    print("Before Calibrating Train Logloss:", log_loss(y_train, y_pred_train))
    print("Before Calibrating Test  Logloss:", log_loss(y_cal, y_pred_cal))

    # キャリブレーション
    y_pred_cal_raw = model.predict_proba(X_cal[feature_col], raw_score=True)
    y_pred_test_raw = model.predict_proba(X_test[feature_col], raw_score=True)
    
    T = temperature_scaling(y_pred_cal_raw, y_cal)

    y_pred_cal_calibrated = scipy.special.softmax(y_pred_cal_raw / T, axis=1)
    y_pred_test_calibrated = scipy.special.softmax(y_pred_test_raw / T, axis=1)
    # 勝率を予測（ついでに過学習していないかチェック）
    print("Calibrated Train Logloss:", log_loss(y_cal, y_pred_cal_calibrated))
    print("Calibrated Test  Logloss", log_loss(y_test, y_pred_test_calibrated))
    print("↑これらは正規化する前のデータを用いていることに注意")

    # Breier_scoreも提示したい。


    # モデルの重要度を表示（gain）
    importances = model.booster_.feature_importance(importance_type="gain")
    feature_name = pd.Series(model.feature_name_)
    indices = np.argsort(importances)[::-1] 
    plt.figure(figsize=(40,6))
    plt.title("Feature importances (gain)")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), feature_name[indices], rotation=90, fontsize="xx-small")
    plt.show()
    # モデルの重要度を表示（split)
    importances = model.booster_.feature_importance(importance_type="split")
    feature_name = pd.Series(model.feature_name_)
    indices = np.argsort(importances)[::-1] 
    plt.figure(figsize=(40,6))
    plt.title("Feature importances (split)")
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), feature_name[indices], rotation=90, fontsize="xx-small")
    plt.show()


    # テストデータで各horse_Nごとのloglossを計算
    # X_testとy_pred_test_calibratedを結合
    prob_calcurated_df = prob_calculator_multiclass(X_test, y_pred_test_calibrated)
    prob_calcurated_df = prob_calcurated_df.set_index(X_test.index) 
    logloss_of_each_horse_N = []

    target_name = [f"pred_{j}_row" for j in range(4)]
    for i in range(5, 19):
        horse_N_index = X_test.horse_N == i
        prob_calcurate_horse_N = prob_calcurated_df.loc[horse_N_index, target_name]
        try:
            logloss_score = log_loss(y_test[horse_N_index], prob_calcurate_horse_N)
            logloss_of_each_horse_N.append(tuple([i, logloss_score]))
        except:
            # レースが存在しない場合はnp.nanを代入
            logloss_of_each_horse_N.append(tuple([i, np.nan]))

    # データを書き込み
    if save_result:
        current_data_dict = dict() # ここにデータを追加していく
        # 登録時間
        now = time.ctime()
        cnvtime = time.strptime(now)
        current_data_dict["date"] = time.strftime("%Y/%m/%d %H:%M", cnvtime)
        # メモ
        current_data_dict["memo"] = memo
        # loglossスコア
        for i, logloss_score in logloss_of_each_horse_N:
            current_data_dict[f"horse_{i}"] = logloss_score
        all_logloss = log_loss(y_true=y_test, y_pred=prob_calcurated_df[target_name])
        current_data_dict["all_logloss"] = all_logloss
        print("logloss is saved")
        # グローバルAUC(正規化前の確率を計算)、ovrを計算
        auc_score_ovr = roc_auc_score(y_test, y_pred_test_calibrated, multi_class="ovr") 
        current_data_dict["auc(ovr)"] = auc_score_ovr
        # 後で、正規化後のaucも加えたい。
        print("auc_score is saved")

        try:
            old_scores = pd.read_csv(scores_path) 
        except:
            score_table_cols = ["date", "memo"] + [f"horse_{i}" for i in range(5, 19)] + ["all_logloss", "auc(ovr)"]
            print("指定されたパスにファイルがなかったので、新しくファイルを生成しました。")
            old_scores = pd.DataFrame(columns=score_table_cols)
        current_score_df = pd.DataFrame([current_data_dict])
        update_scores = pd.concat([current_score_df, old_scores], axis=0)
        update_scores.to_csv(scores_path, index=False)
        display(update_scores.head(5))

    # 返すデータの設定（予測値を埋め込む）
    X_test = prob_calculator_multiclass(X_test, y_pred_test_calibrated)
    X_test.loc[:, "target"] = y_test.values

    return model, X_test


# objective関数を作る関数
def create_objective(X, y, splitter, feature_col, params):
    def objective(trial):
        # パラメータの定義
        tuning_params = {
            "max_bin": trial.suggest_int("max_bin", 10, 255),
            "num_leaves": trial.suggest_int("num_leaves", 2, 100),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-8, 10),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.8),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 100.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
            "max_depth": trial.suggest_int("max_depth", 2, 100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "path_smooth": trial.suggest_float("path_smooth", 1, 5),
            "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.6, 0.8),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False])
        }
        # lightGBMに渡すパラメータ
        kwargs = {**params, **tuning_params}

        # 交差検証
        oof_preds = np.full((len(X), 4), np.nan)
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(X, y, groups="id_for_fold")):

            X_train, y_train = X.iloc[tr_idx, :], y.iloc[tr_idx]
            X_test, y_test = X.iloc[val_idx, :], y.iloc[val_idx]

            # 要らない列を削除
            X_train, X_test = X_train[feature_col], X_test[feature_col]

            # modelの宣誓とコールバックの定義
            model = LGBMClassifier(**kwargs)
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                LightGBMPruningCallback(trial, "multi_logloss")
                ]
             
            # modelの学習
            model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      eval_metric="multi_logloss",
                      callbacks=callbacks)

            # 予測値の取得
            oof_preds[val_idx] = model.predict_proba(X_test)

        # NaN ではない部分のインデックスを取得
        not_nan_indices = ~np.isnan(oof_preds).any(axis=1)

        # NaN ではない部分のみでmulti_logloss を計算
        # レース内で正規化する前のmulti_loglossを計算する
        avg_logloss = log_loss(y[not_nan_indices], oof_preds[not_nan_indices])

        return avg_logloss

    return objective


# 予測確率を正規化する関数
def prob_calculator_multiclass(
        df_to_copy: pd.DataFrame,
        prob: np.ndarray,
        class_names=None,
        group_col: str = "id_for_fold",
        epsilon: float = 1e-15,
) -> pd.DataFrame:
    """
    レース単位（group_col ごと）で *クラスごとの確率総和＝1* になるよう正規化する。

    Parameters
    ----------
    df_to_copy : pd.DataFrame
        馬ごとの元データ（行 = 馬）。
    prob : np.ndarray, shape = (n_samples, n_classes)
        LightGBM などが返す確率行列（softmax 済み or 温度スケール後など）。
    class_names : list[str] | None
        列名に用いるクラス名。None の場合は ["class_0", "class_1", …] を自動生成。
    group_col : str
        レース ID を示す列名。デフォルトは "id_for_fold"。
    epsilon : float
        0 や 1 で LogLoss が発散しないようにするクリップ値。

    Returns
    -------
    pd.DataFrame
        元データに `pred_{class}` 列を追加した DataFrame。
        *レース単位* で各クラス確率の合計が 1 になる。
    """
    df = df_to_copy.copy()

    # --- 0. クラス名を決定 ---
    n_classes = prob.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    # --- 1. クリッピング（数値安定化） ---
    prob_clip = np.clip(prob, epsilon, 1.0 - epsilon)

    # --- 2. レース単位で正規化 ---
    for j, cls in enumerate(class_names):
        if cls == "class_0":
            continue
        col_pred = f"pred_{cls}"
        df[col_pred] = prob_clip[:, j]

        # レース内のクラス確率総和
        race_sum = df.groupby(group_col, observed=False)[col_pred].transform("sum")
        df[col_pred] /= race_sum          # レース単位で合計 = 1 へ再スケール

    # --- 3. 行単位で正規化 ---
    row_sum = prob_clip.sum(axis=1)
    for i in range(n_classes):
        prob_name = f"pred_{i}_row"
        df[prob_name] = prob_clip[:, i] / row_sum

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


# temperature_scaling
def temperature_scaling(logits_val, y_val):
    """
    logits_val : shape (N, C)  : LightGBM の predict(before_softmax=True) 相当
    y_val      : shape (N,)    : 0〜C-1
    """
    # 負の対数尤度を最小化する温度 T を求める
    def nll(T):
        T = T[0]
        # ソフトマックス
        proba = scipy.special.softmax(logits_val / T, axis=1)
        return log_loss(y_val, proba)

    best_T = minimize(nll, x0=[1.0], bounds=[(0.05, 10)]).x[0]
    return best_T