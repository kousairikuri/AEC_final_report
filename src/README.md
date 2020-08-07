Implementation of ThermoDynamical Genetic Algorithm (TDGA) in Python

[遺伝アルゴリズムにおける熱力学的選択ルールの提案](https://www.jstage.jst.go.jp/article/iscie1988/9/2/9_2_82/_pdf)
___
## Requirements
deap >= 1.3.1

## Usage
deap の使い方がわかっている人向け．

deap を触ったことがない人はまず[チュートリアル](https://deap.readthedocs.io/en/master/overview.html)から．

#### tools.registerで選択演算として登録
```python
from tdga.td_selection import ThermoDynamicalSelection

Np = 32
t_init = 10
tds = ThermoDynamicalSelection(Np=Np, t_init=t_init, scheduler=lambda x: x)
toolbox.register("select", tds.select)
```
#### 指定必須パラメータ
- Np: 個体数
- t_init: 初期温度

#### 追加パラメータ
- Ngen: 終了世代. スケジューラを指定しない場合必要.
- t_fin: 終了温度. スケジューラを指定しない場合必要.
- scheduler: 温度スケジューラ. 関数オブジェクト.

ex) 恒等スケジューラ
```python
scheduler = lambda x: x
```
- is_compress: 個体群圧縮用のフラグ. 同一の遺伝子型を持つ個体を削除. 熱力学的選択では同一の個体が複数あっても結果は同じ. デフォルト: False

#### スケジューラ
スケジューラを指定しない場合，デフォルトでの更新式は
```
t = gen_current/Ngen
T = t_init^(1-t) * t_fin^t
```
が採用されます. この場合, t_finとNgenを指定してください．ただし, t_fin に 0 は設定できません.


#### 分析ツール
各世代の個体や統計量を保存して分析するクラスです. analyzer.py に実装されています.

よしなに拡張して使ってください.

1. Analyzer インスタンスを初期化
2. 各世代の pop を add_pop() で追加. 加える世代は

```list(map(toolbox.clone, pop))```
のようにしてメモリ参照を切ってください.

#### plot_entropy_matrix() でグラフプロット
```
analizer.plot_entropy_matrix(file_name="entropy.png")
```
#### plot_stats() で統計量をグラフプロット.
最適解の指定は
```python
analizer.plot_stats(file_name="stats.png", optimum_val=1099)
```
のように指定可能です.

### ナップサック問題適用例
TDGA/tests/benchmark_knapsack.py を参照.

## 開発者のためのTIPS
#### 遺伝子の持つ情報へのアクセス

individual は
```python
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
```
によって生成されるIndividualクラス．

インデクシングやgetattrなどで遺伝子配列や適応度情報にアクセスできる．

ex)
```python
ind = pop[0]
print(ind)
>> [1, 0, 1, 1, 0, 0, 0, 1]
print(ind[0])
>> 1
print(getattr(ind, "fitness").values[0])  # 適応度へのアクセス
>> 780
print(ind.fitness.values[0])  # 上と同じ
>> 780
print(ind.fitness.wvalues[0])  # 重み付けされた適応度
>> 780
```

### できていること
- 論文内容の実装
- 対立遺伝子の整数拡張
- 最小化問題への適用

### やっていないこと
- 例外処理, 単体テスト
- 実数値最適化問題への適用
- 多目的最適化問題への適用
- TDGA 収束の解決諸手法の実装
- 厳密な実装における最適化(高速化)
- TSP への適用

> TDGA の TSP への適用は枝のエントロピーを最大化する手法が有効である. このとき管理すべきは
頂点 i から頂点 j に繋がれているような枝の数である.
