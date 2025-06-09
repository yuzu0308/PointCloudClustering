import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

print("-> Open3Dのサンプルデータを読み込みます。")

sample_data = o3d.data.DemoICPPointClouds()
pcd_path = sample_data.paths[1]
pcd = o3d.io.read_point_cloud(pcd_path)
o3d.visualization.draw_geometries([pcd], window_name="Original PointClouds")

# ダウンサンプリング
voxel_size = 0.02 
pcd_down = pcd.voxel_down_sample(voxel_size)
print(f"-> ダウンサンプリング後の点数: {len(pcd_down.points)}")

eps = 0.05
min_points = 10

print(f"-> DBSCANクラスタリングを実行します (eps={eps}, min_points={min_points})")
# cluster_dbscanは各点のクラスタIDをNumpy配列で返す（-1はノイズ）
labels = np.array(pcd_down.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

# クラスタリング結果を可視化
max_label = labels.max()  
print(f"-> {max_label + 1}個のクラスタが検出されました。")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = [0.5, 0.5, 0.5, 1]  

pcd_down.colors = o3d.utility.Vector3dVector(colors[:, :3])  

print("-> クラスタリング結果を表示します。")
o3d.visualization.draw_geometries([pcd_down], window_name="DBSCAN Clustering")
