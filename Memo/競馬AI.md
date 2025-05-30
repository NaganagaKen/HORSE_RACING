# 競馬AIメモ

## 特徴量

| 変数名(変換前)         | 変数名(変換後)       | 型     | 説明                                                                                                                                  | 欠損            |
|------------------------|----------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| レースID(新)           | race_id              | int    | yearを作成するために取り込み +00～03:年(4byte), +04,05:月, +06,07:日, +08,09:場所コード, +10,11:回次, +12,13:日次, +14,15:レース番号, +16,17:馬番号 | なし            |
| 年                     | year                 | int    | 2桁表記をrace_idを使用して4桁に上書き                                                                                               | なし            |
| 月                     | month                | int    | 月                                                                                                                                    | なし            |
| 日                     | day                  | int    | 日                                                                                                                                    | なし            |
| 回次                   | times                | int    | 何回目の開催か                                                                                                                        | なし            |
| 場所                   | place                | chr    | 開催場所                                                                                                                              | なし            |
| 日次                   | daily                | chr    | 開催の何日目か(1,2,3,…,9,A,B,C)                                                                                                       | なし            |
| レース番号             | race_num             | int    | 何レース目か                                                                                                                          | なし            |
| 馬名                   | horse                | chr    | 馬名, 馬のid代わり                                                                                                                    | なし            |
| 騎手コード             | jockey_id            | int    | 騎手id                                                                                                                                | なし            |
| 頭数                   | horse_N              | int    | 出馬数                                                                                                                                | なし            |
| 枠番                   | waku_num             | int    | 枠番                                                                                                                                  | なし            |
| 馬番                   | horse_num            | int    | 馬番                                                                                                                                  | なし            |
| クラスコード           | class_code           | int    | レースの種類（G1など）                                                                                                                | なし            |
| トラックコード(JV)    | track_code           | int    | 後でトラックコードから変換                                                                                                            | なし            |
| コーナー回数           | corner_num           | int    | コースのコーナー回数                                                                                                                  | 欠損率3.3%、2で埋め |
| 距離                   | dist                 | int    | コースの距離                                                                                                                          | なし            |
| 馬場状態               | state                | chr    | コースの状態                                                                                                                          | なし            |
| 天候                   | weather              | chr    | 天候                                                                                                                                | なし            |
| 年齢限定(競走種別コード) | age_code             | int    | 競走種別コード（詳細不明）                                                                                                            | なし            |
| 性別                   | sex                  | chr    | 牡、牝、騙馬                                                                                                                          | なし            |
| 年齢                   | age                  | int    | 年齢                                                                                                                                | なし            |
| 斤量                   | basis_weight         | num    | 出走時の負担重量                                                                                                                      | なし            |
| ブリンカー             | blinker              | chr    | ブリンカーフラグ                                                                                                                      | なし            |
| 馬体重                 | weight               | int    | 馬体重                                                                                                                              | 欠損率0.26%、平均埋め |
| 増減                   | inc_dec              | int    | 馬体重の増減                                                                                                                          | 欠損率9.1%、0埋め |
| 重量コード             | weight_code          | int    | 重量負荷種別                                                                                                                          | なし            |
| 単勝オッズ             | win_odds             | num    | 確定単勝オッズ                                                                                                                        | —               |
| 確定着順               | rank                 | int    | 確定順位                                                                                                                              | なし            |
| 着差タイム             | time_diff            | chr    | 「---」は取消など、1着は2着とのタイム差をマイナス値                                                                                 | なし            |
| 走破タイム(秒)         | time                 | num    | 走破タイム                                                                                                                            | なし            |
| 通過順1角              | corner1_rank         | int    | 1コーナー順位、0は欠損                                                                                                                | 0は使用しない   |
| 通過順2角              | corner2_rank         | int    | 2コーナー順位、0は欠損                                                                                                                | 0は使用しない   |
| 通過順3角              | corner3_rank         | int    | 3コーナー順位、0は欠損                                                                                                                | 0は使用しない   |
| 通過順4角              | corner4_rank         | int    | 4コーナー順位、0は欠損                                                                                                                | 0は使用しない   |
| 上がり3Fタイム         | last_3F_time         | num    | 最後600mの走破タイム                                                                                                                  | 欠損率0.9%      |
| 上がり3F順位           | last_3F_rank         | int    | 最後600mでの順位                                                                                                                      | 欠損率3.4%      |
| Ave-3F                | Ave_3F               | num    | ラスト3Fまでの平均タイム                                                                                                              | 欠損率4%        |
| PCI                   | PCI                  | num    | ペースチェンジ指数                                                                                                                    | 欠損率4%        |
| -3F差                 | last_3F_time_diff    | num    | ゴール前3Fのタイム差                                                                                                                  | 欠損率4%        |
| 脚質                   | leg                  | chr    | レースの走り方                                                                                                                        | なし            |
| 人気                   | pop                  | int    | 人気順位（error_code = 0では欠損なし）                                                                                                | —               |
| 賞金                   | prize                | int    | 獲得賞金                                                                                                                              | なし            |
| 異常コード             | error_code           | int    | 0=正常, 1=出走取消, 2=発送除外, 3=競走除外, 4=競走中止, 5=失格, 6=落馬再騎乗, 7=降着                                                 | なし            |
| 父馬名                 | father               | chr    | 父馬名                                                                                                                                | なし            |
| 母馬名                 | mother               | chr    | 母馬名                                                                                                                                | なし            |
| 血統登録番号           | id                   | int    | 馬id（10桁）                                                                                                                           | なし            |


## EDA で気になったこと
- レース前に知ることが出来ない情報がある。
  - 着差タイム
  - 走破タイム
  - 通過順1-4角
  - 上がり3Fタイム
  - 上がり3F順位
  - Ave-3F
  - PCI
  - -3F差
  - 獲得賞金
  - win_odds (これは馬券売買終了後に確定するため、データリークとなり得る)
- blinkerの欠損値が多すぎる。   
  - どうやらブリンカーを付けていない馬は値が入力されていないらしい。
- モデルを二段階に分ける
  - 1着になる確率を予測 (機械学習で予測)
  - 1着になる確率とオッズから期待収支を計算する。（アルゴリズムを作る）


### EDAメモ
#### 後で見る事
- 月ごとにどのような階級のレースが行われているのか
- 年月日から何曜日開催が多いのか
- 各horse_Nでどのようなレース種別が多いのか
- 各horse_Nはどの月に多いのか
- 距離と競馬場の組み合わせがどこなのか調べる
- horse_Nが小さいレースはどのようなものがあるか（どのような種別？）
  - オッズの付き方に特徴があるか調べる。
- レース前に分かる情報は何か調べる。
- あとでデータサイエンス研究会のdriveにあるサンプルを見てみる。


#### 疑問点
- jockey_idはどういう法則で番号を付けているのか分からない。
- class_codeがなぜ数値が飛び飛びの値なのか？
- error_codeの取り扱い
  - 1のデータは使用せずに学習させる？
- 脚質のコードがよくわからない
  - 中団・後方・マクリだけ中盤での傾向を指している。



## これからやること
- バリテーションの方法をもう一度確認
  - まず全データを訓練データとテストデータで分ける。
  - そして、訓練データをクロスバリデーションを行いパラメータチューニングを行う。
  - 最後にテストデータを用いてデータを能力を計測する。

- 特徴量エンジニアリング

- 明らかに障害レースと通常のレースはデータの取り扱い方が違うので、モデルを分ける必要がある。

- ドメイン知識の収集
  - プロの馬券師や競馬新聞はどのようにしてマークを付けているのかを調べる。　
  - 競馬場、距離、枠番号で傾向がある可能性あり。過去データから傾向を調べる。

- 欠損値の意味をよく考える。
  - 欠損地そのものが予測に影響を与える可能性がある。（出走取消）

- 当日情報が簡単に手に入らないものは特徴量として考えない。
  - 体重・体重増減など？

- loglossを頭数ごとに出力する？或いはモデルを分ける？
  - 障害、ダート、G1, G2, G3 ,...などでも分けてみる

- 期待値計算、馬券購入の戦略は後回し。
  - ポートフォリオを組むことになるか。


### 特徴量エンジニアリング（案）
- weightとbasis_weightを足す
- weightとbasis_weightの比率
- ageを2,3,over4としてカテゴリ化
- blinkerの欠損値埋め（付けていない馬を"N"とかで埋める）
- time_diffを処理しないといけない
  - 特に取り消しをどう処理するべきか調べる
- 中何週か（日時と馬の名前から計算できる）
- ジョッキーの過去の勝率を特徴量として加える。（勝率も2通り存在？）
- 過去、同じmother, fatherの馬がどのような成績だったのかを特徴量として加える。（賞金等は頭数で割って正規化する）
- 過去のレースのlast_3F_time, PCIがどのくらいだったか。
  - それと各馬の過去3レースのlast_3F_timeの平均を比較する
    - ただし、距離(ビン分割が必要そう)とコース（芝・ダート、直線）によって分ける。
- 体重変化率
- 平均オッズ
- 平均斤量
- 各ジョッキーに対して、過去どの程度違反（失格、降着、落馬再騎乗）があったかをジョッキーの特徴量として加える。
- targetの作り方に注意（rankが0は欠損値（異常終了））
- Elo rating?
- TrueSkill?

**こんな感じで作った特徴量をPCAなどでまとめる**
- 特徴量は、馬の特徴量(血統、過去のデータ)、ジョッキーの特徴量は分ける。


### うまく行かなかった特徴量エンジニアリング
- mother, fatherなどの重要そうなカテゴリデータは全部繋げてtf-idfでベクトル化する（bertでもやってみる）


## 定例会メモ
- 5/8 Thu.
  - 確率ならloglossを使って検証する。
  - 1着かそれ以外かで2値分類
    - 頭数で確率が大きく変わるので、工夫が必要。頭数を使って考える。
  - 単勝の確定オッズを予測して、それを使って戦略を立てたい。（TimeSeriesOddsから予測する） 
    - 確定オッズは馬券販売終了後に確定するからリークになってしまう。
  - LightGBMは確率として適当なので、LightGBMで出た確率をロジスティック回帰で滑らかにするといいかも。
  - 最適レートについて調べる。（馬券の購入方法として重要らしい）

5/15
  - EDAをちゃんとやるべき

5/22
  - EDAをちゃんとやったので、モデルを作成
