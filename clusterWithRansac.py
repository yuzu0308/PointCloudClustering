import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def main():
    # --- 1. データの準備 ---
    print("-> 1. サンプルデータを読み込みます。")
    pcd_path = o3d.data.DemoICPPointClouds().paths[1]
    pcd = o3d.io.read_point_cloud(pcd_path)

    # 処理前の点群
    pcd_original = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd_original], window_name="Original Pointclouds")

    geometries = [] # 最終的に可視化するジオメトリのリスト

    # --- 2. 床(平面)の検出 (RANSAC) ---
    print("-> 2. 床を検出します。")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    
    # [a, b, c, d]：平面の方程式 ax + by + cz + d = 0 の係数
    # 法線ベクトルは (a, b, c)
    [a, b, c, d] = plane_model
    print(f"   平面モデル: a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")

    # 法線ベクトルのZ成分の絶対値が1に近い -> 水平な面（床または天井）
    if abs(c) > 0.9:
        print("   -> 床を検出しました。")
        # inliersは平面上にある点のインデックス
        floor = pcd.select_by_index(inliers)
        floor.paint_uniform_color([0.0, 0.0, 1.0]) # 床を青色に
        geometries.append(floor)

        # 残りの点群（床以外）
        pcd = pcd.select_by_index(inliers, invert=True)
    else:
        print("   -> 明確な床は検出できませんでした。")
    
    # --- 3. 壁の検出 (RANSAC) ---
    print("-> 3. 壁を検出します。")
    num_walls = 4 # 最大で2つの壁を検出
    all_wall_inliers = []
    for i in range(num_walls):
        # 残りの点群からさらに平面を検出
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        [a, b, c, d] = plane_model

        # 法線ベクトルのZ成分の絶対値が0に近い -> 垂直な面（壁）
        if abs(c) < 0.2:
            print(f"   -> 壁 {i+1} を検出しました。")
            all_wall_inliers.extend(inliers) # 壁のインデックスを保存
            
            # 次の壁検出のために、今見つけた壁を点群から取り除く
            pcd = pcd.select_by_index(inliers, invert=True)

    if all_wall_inliers:
        # 保存したインデックスを使って、元の点群から壁全体を抽出
        walls = pcd_original.select_by_index(all_wall_inliers)
        walls.paint_uniform_color([0.0, 1.0, 0.0]) # 壁を緑色に
        geometries.append(walls)
    
    # --- 4. & 5. その他オブジェクトのクラスタリング (DBSCAN) ---
    print("-> 4. 残りのオブジェクトをDBSCANでクラスタリングします。")
    # この時点でpcdには床でも壁でもないオブジェクトのみが残っている
    remaining_pcd = pcd
    labels = np.array(remaining_pcd.cluster_dbscan(eps=0.03, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"   -> {max_label + 1}個のオブジェクトクラスタを検出しました。")

    # クラスタごとに色分け
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = [0.5, 0.5, 0.5, 1]  # ノイズは灰色
    remaining_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    geometries.append(remaining_pcd)


    # --- 6. 統合して可視化 ---
    print("-> 6. すべての要素を統合して表示します。")
    print("   青: 床, 緑: 壁, その他: オブジェクト")
    o3d.visualization.draw_geometries(geometries, window_name="Floor, Wall, and Object Segmentation")


if __name__ == "__main__":
    main()