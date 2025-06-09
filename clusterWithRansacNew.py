import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def main():
    # --- パラメータ設定 ---
    # RANSAC用パラメータ
    RANSAC_DISTANCE_THRESHOLD = 0.015 # 平面とみなす距離のしきい値
    MIN_POINTS_FOR_PLANE = 1000     # 平面として検出するのに必要な最小点数
    PLANE_COUNTS_MAX = 4 #検出する平面の最大数

    # DBSCAN用パラメータ
    DBSCAN_EPS = 0.04
    DBSCAN_MIN_POINTS = 15

    # --- 1. データの準備 ---
    print("-> 1. サンプルデータを読み込みます。")
    pcd_path = o3d.data.DemoICPPointClouds().paths[0]
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # 可視化用に処理前の点群をコピーしておく
    pcd_original = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd_original], window_name="Original Pointclouds")

    # 残りの点群を処理対象とする
    remaining_pcd = pcd
    
    # 検出したジオメトリを格納するリスト
    floor_segments = []
    wall_segments = []
    
    plane_counts = 0
    # --- 2. 反復的な平面検出と分類 ---
    print("-> 2. 平面を反復的に検出し、床と壁に分類します。")
    while len(remaining_pcd.points) > MIN_POINTS_FOR_PLANE:
        # a. 残っている点群から最も大きな平面を1つ検出
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=RANSAC_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=1000
        )
        
        # 大きな平面が見つからなければループを抜ける
        if len(inliers) < MIN_POINTS_FOR_PLANE:
            break

        plane_counts += 1
        if plane_counts > PLANE_COUNTS_MAX:
            break
        # b. 検出した平面の「向き」で分類
        [a, b, c, d] = plane_model
        
        # 検出した平面の点群を取得
        plane_segment = remaining_pcd.select_by_index(inliers)
        
        if abs(c) > 0.9: # 法線のZ成分が1に近い -> 水平面（床）
            print(f"  -> 床(水平面)を検出しました ({len(inliers)}点)。")
            floor_segments.append(plane_segment)
        elif abs(c) < 0.2: # 法線のZ成分が0に近い -> 垂直面（壁）
            print(f"  -> 壁(垂直面)を検出しました ({len(inliers)}点)。")
            wall_segments.append(plane_segment)
        
        # d. 検出した平面を残りの点群から取り除く
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    print("-> 平面の検出が完了しました。")

    # --- 3. 平面の統合と着色 ---
    geometries = []
    if floor_segments:
        floor = o3d.geometry.PointCloud()
        for segment in floor_segments:
            floor += segment
        floor.paint_uniform_color([0.0, 0.0, 1.0]) # 床を青色に
        geometries.append(floor)

    if wall_segments:
        walls = o3d.geometry.PointCloud()
        for segment in wall_segments:
            walls += segment
        walls.paint_uniform_color([0.0, 1.0, 0.0]) # 壁を緑色に
        geometries.append(walls)


    # --- 4. オブジェクトのクラスタリング ---
    print("-> 4. 残りのオブジェクトをDBSCANでクラスタリングします。")
    if len(remaining_pcd.points) > 0:
        labels = np.array(remaining_pcd.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=True
        ))
        max_label = labels.max()
        print(f"   -> {max_label + 1}個のオブジェクトクラスタを検出しました。")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = [0.5, 0.5, 0.5, 1]
        remaining_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        geometries.append(remaining_pcd)
    else:
        print("   -> クラスタリング対象のオブジェクトはありませんでした。")

    # --- 5. 最終的な可視化 ---
    print("-> 5. すべての要素を統合して表示します。")
    print("   青: 床, 緑: 壁, その他: オブジェクト")
    o3d.visualization.draw_geometries(geometries, window_name="Unified Plane Detection and Clustering")


if __name__ == "__main__":
    main()