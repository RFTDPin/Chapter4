import cv2
import numpy as np


def draw_trajectories(image, trajectories):
    """
    根据存储历史轨迹的字典来高效绘图。
    :param image: 要绘制的图像帧
    :param trajectories: 轨迹字典, {mmsi: [(x1, y1), (x2, y2), ...]}
    """
    overlay = image.copy()

    # 设置字体和线条粗细
    tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)

    for mmsi, points in trajectories.items():
        # 将点列表转换为numpy数组，方便使用polylines函数一次性绘制所有线段
        if len(points) > 1:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=False, color=(255, 0, 0), thickness=3)

        # 只在最新的点上绘制MMSI标签
        if points:
            last_point = points[-1]
            label = f'MMSI:{int(mmsi)}'
            cv2.putText(overlay, label, (last_point[0], last_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, tl / 3, [0, 0, 255],
                        thickness=tf, lineType=cv2.LINE_AA)

    # 可以使用cv2.addWeighted来创建带有透明效果的轨迹
    # alpha = 0.7
    # return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return overlay