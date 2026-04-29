import cv2
import numpy as np
import argparse
import os

def detect_and_link_limited_holes(image_path, output_path, dist_threshold=32, offset_threshold=2.5):
    # 讀取與影像預處理
    img = cv2.imread(image_path)
    if img is None: return
    display_img = img.copy()
    
    contrast_img = np.clip(display_img * (80/127 + 1) - 80, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(contrast_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    bw_display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 提取特徵並過濾直徑 (10~50px) 與圓度
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hole_centers = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        diameter = radius * 2
        if 6 < diameter < 16:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.45:
                    hole_centers.append((int(x), int(y)))

    # 建立所有可能的連線對並計算距離
    possible_pairs = []
    num_holes = len(hole_centers)
    for i in range(num_holes):
        for j in range(i + 1, num_holes):
            p1 = hole_centers[i]
            p2 = hole_centers[j]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            
            if dist <= dist_threshold and dist >= 15:
                possible_pairs.append({
                    'points': (i, j),
                    'distance': dist
                })

    # 按距離排序並限制連線數，建立「鄰接串列」
    possible_pairs.sort(key=lambda x: x['distance'])
    
    # 用來記錄每個點連接了哪些鄰居 (Adjacency List)
    adjacency_list = {i: [] for i in range(num_holes)}
    
    for pair in possible_pairs:
        idx1, idx2 = pair['points']
        
        # 檢查兩個端點是否都還沒超過連線上限 (4條)
        if len(adjacency_list[idx1]) < 5 and len(adjacency_list[idx2]) < 5:
            p1 = hole_centers[idx1]
            p2 = hole_centers[idx2]
            
            cv2.line(display_img, p1, p2, (0, 0, 255), 2)
            cv2.line(bw_display_img, p1, p2, (0, 0, 255), 2)
            
            # 記錄雙向連結關係
            adjacency_list[idx1].append(idx2)
            adjacency_list[idx2].append(idx1)

    # 小偏移瑕疵偵測 + 轉角過濾 (Local Collinearity & Angle Check)
    defect_count = 0
    for i, neighbors in adjacency_list.items():
        # 只有在節點位於網格內部邊緣（有兩個鄰居）時才進行判定
        if len(neighbors) == 2:
            p0 = hole_centers[i]
            p1 = hole_centers[neighbors[0]]
            p2 = hole_centers[neighbors[1]]
            
            # 建立向量 v1 與 v2
            v1 = (p1[0] - p0[0], p1[1] - p0[1])
            v2 = (p2[0] - p0[0], p2[1] - p0[1])
            
            # 計算內積與向量長度
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = np.hypot(v1[0], v1[1])
            mag2 = np.hypot(v2[0], v2[1])
            
            if mag1 > 0 and mag2 > 0:
                cos_theta = dot_product / (mag1 * mag2)
                
                # 夾角接近 180 度 (cos_theta < -0.85) 才視為「應為直線」
                if cos_theta < -0.85:
                    # 向量外積法計算 P0 到線段 P1-P2 的垂直距離
                    num = abs((p0[0] - p1[0]) * (p2[1] - p1[1]) - (p0[1] - p1[1]) * (p2[0] - p1[0]))
                    den = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    
                    if den > 0:
                        offset_dist = num / den
                        # 若偏離基準線大於設定閾值
                        if offset_dist > 3:
                            defect_count += 1
                            cv2.circle(display_img, p0, 12, (0, 165, 255), 3)
                            cv2.circle(bw_display_img, p0, 12, (0, 165, 255), 3)
                            cv2.putText(display_img, f"NG:{offset_dist:.1f}", (p0[0]+15, p0[1]-15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    # 畫出正常中心點以利觀察
    for center in hole_centers:
        cv2.circle(display_img, center, 4, (255, 255, 0), -1)
        cv2.circle(bw_display_img, center, 4, (255, 255, 0), -1)

    # 儲存
    cv2.imwrite(output_path, display_img)
    base_name, ext = os.path.splitext(output_path)
    cv2.imwrite(f"{base_name}_bw{ext}", bw_display_img)
    
    print(f"檢測完成：共找到 {num_holes} 個孔，發現 {defect_count} 處異常偏移。")

# 可調整 offset_threshold 來定義多大的偏移算 NG
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_img", type=str, required=True)
    args = parser.parse_args()
    detect_and_link_limited_holes(args.input_img, 'result_defect.jpg', dist_threshold=32, offset_threshold=2.5)