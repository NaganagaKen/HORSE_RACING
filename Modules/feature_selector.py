from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from IPython.display import display
import lightgbm as lgb


# lightGBMから特徴量を選択する関数
def select_from_model(df, feature_col):
    df = df.copy()
    feature_col = feature_col.copy()
    
    # 2年分のみのデータを用いて特徴量選択を行う。 
    unique_years = sorted(df["year"].unique())
    select_df = df[df["year"] <= unique_years[1]]
    ret_df = df[df["year"] > unique_years[1]]

    select_df = select_df.loc[:, ~select_df.columns.duplicated()]
    feature_col = list(dict.fromkeys(feature_col))            # 重複名を削除
    feature_col = [c for c in feature_col if c in select_df.columns]

    cat_col = select_df[feature_col].select_dtypes(include=["object"]).columns.tolist()
    select_df[cat_col] = select_df[cat_col].astype("category")

    # モデルの定義
    params = {
              # 確定パラメータ
              "objective" : "multiclass",
              "metric" : "multi_logloss",
              "n_estimators": 5000,
              "n_jobs" : -1,
              "verbose" : -1,
              "random_state" : 42,
              # 調整したパラメータ
              'max_bin': 144, 
              'num_leaves': 20, 
              'min_data_in_leaf': 98, 
              'min_sum_hessian_in_leaf': 7.751328235859817, 
              'bagging_fraction': 0.7757995766256757, 
              'bagging_freq': 90, 
              'feature_fraction': 0.6381099809299766, 
              'lambda_l1': 16.547879518482567, 
              'lambda_l2': 7.672290184186785e-08, 
              'min_gain_to_split': 1.959828624191452, 
              'max_depth': 6, 
              'learning_rate': 0.007551909976018511, 
              'path_smooth': 2.554709158757928, 
              'feature_fraction_bynode': 0.6542698063547792, 
              'extra_trees': True
              }
    
    model = lgb.LGBMClassifier(**params)
    selector = SelectFromModel(model, max_features=200)

    # 学習開始
    X, y = select_df.drop(["target"], axis=1)[feature_col], select_df["target"]
    selector.fit(X, y)

    # 上位の特徴量を抽出
    selected_feature_col = []
    for col, mask in zip(feature_col, selector.get_support()):
        if mask:
            selected_feature_col.append(col)

    return ret_df, selected_feature_col