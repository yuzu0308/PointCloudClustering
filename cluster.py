import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("-> Open3Dのサンプルデータを読み込みます。")
# Open3Dが提供するサンプルデータセット（室内の一部をスキャンしたもの）を使用
# このデータは関数呼び出し時に自動的にダウンロードされます。
sample_data = o3d.data.DemoICPPointClouds()
pcd_path = sample_data.paths[1]
pcd = o3d.io.read_point_cloud(pcd_path)
o3d.visualization.draw_geometries([pcd], window_name="Original PointClouds")
# 元の点群を一度表示してみる（任意）
# print("元の点群を表示します。ウィンドウを閉じて次に進んでください。")
# o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

# ダウンサンプリングで点群の数を減らし、処理を高速化する
voxel_size = 0.02  # 5cmの立方体でダウンサンプリング
pcd_down = pcd.voxel_down_sample(voxel_size)
print(f"-> ダウンサンプリング後の点数: {len(pcd_down.points)}")

# DBSCANクラスタリングを実行
# eps: ある点を核（コア）点とみなすための、近傍点の最大距離。この値がDBSCANの挙動を大きく左右します。
#      小さすぎるとクラスタが細分化されすぎ、大きすぎると一つの巨大なクラスタになりがちです。
#      一般的に、ダウンサンプリングしたvoxel_sizeの1.5~2倍程度から試すのが良いとされます。
# min_points: ある点が核点とみなされるために、eps距離内に必要な点の最小数。
eps = 0.05
min_points = 10

print(f"-> DBSCANクラスタリングを実行します (eps={eps}, min_points={min_points})")
# cluster_dbscanは各点のクラスタIDをNumpy配列で返す（-1はノイズ）
labels = np.array(pcd_down.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

# クラスタリング結果を可視化
max_label = labels.max()  # ノイズ（-1）を除いた最大クラスタIDを取得
print(f"-> {max_label + 1}個のクラスタが検出されました。")

# 各クラスタに異なる色を割り当てる
# matplotlibのカラーマップを利用して、見やすい色のリストを生成
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# ノイズ（-1）と判定された点は灰色にする
colors[labels < 0] = [0.5, 0.5, 0.5, 1]  # RGBA形式なのでA(透明度)も指定

# PointCloudオブジェクトのcolorsプロパティに色情報を設定
pcd_down.colors = o3d.utility.Vector3dVector(colors[:, :3])  # RGBのみ使用

# 可視化
print("-> クラスタリング結果を表示します。")
o3d.visualization.draw_geometries([pcd_down], window_name="DBSCAN Clustering")
