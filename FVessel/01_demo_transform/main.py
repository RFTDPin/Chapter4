import os
import time
import imutils
import cv2
import argparse
import pandas as pd
import numpy as np

# 只导入需要的、重构后的函数
from utils.file_read import read_all, update_time
from utils.AIS_utils import transform_ais_for_frame, preload_all_ais_data
from utils.draw_org import draw_trajectories


def main(arg):
    # 1. 预加载所有AIS数据到内存
    print("Preloading all AIS data, please wait...")
    ais_data_map = preload_all_ais_data(arg.ais_path)
    print(f"Loaded data for {len(ais_data_map)} timestamps.")

    # 2. 视频和其他参数初始化
    cap = cv2.VideoCapture(arg.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {arg.video_path}")
        return

    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    im_shape = [im_width, im_height]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        print("Warning: Video FPS is 0, setting to 25 as a default.")
        fps = 25
    frame_duration_ms = int(1000 / fps)

    ship_trajectories = {}

    # 视频写入器初始化
    show_size = 1000
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_h = show_size
    out_w = int(im_width * (show_size / im_height)) if im_height > 0 else 0
    videoWriter = cv2.VideoWriter(arg.result_video, fourcc, fps, (out_w, out_h))

    # 时间初始化
    current_time = arg.initial_time
    print(f'Start Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")} || fps: {fps}')

    proc_times = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        current_time, timestamp_ms, time_key = update_time(current_time, frame_duration_ms)

        current_ais_df = ais_data_map.get(time_key, pd.DataFrame())

        if not current_ais_df.empty:
            transformed_points = transform_ais_for_frame(current_ais_df, arg.camera_para, im_shape)

            for mmsi, x, y in transformed_points:
                if mmsi not in ship_trajectories:
                    ship_trajectories[mmsi] = []
                ship_trajectories[mmsi].append((x, y))

        proc_time = time.time() - start_time
        proc_times.append(proc_time)

        frame = draw_trajectories(frame, ship_trajectories)

        frame_count += 1
        # 确保fps不为0
        if fps > 0 and frame_count % fps == 0:
            avg_time = np.mean(proc_times)
            std_time = np.std(proc_times)
            # 【修正】: 使用 timestamp_ms 变量
            print(
                f'Time: {pd.to_datetime(timestamp_ms, unit="ms")} | Avg Process Time: {avg_time:.6f} +- {std_time:.6f} s | FPS: {1 / avg_time if avg_time > 0 else 0:.2f}')
            proc_times = []

        result = imutils.resize(frame, height=show_size)
        videoWriter.write(result)
        # cv2.imshow('demo', result)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    print("Processing finished.")


if __name__ == '__main__':
    # 注意，这里的clip-%02d可能需要根据你的实际目录调整
    path = './data/clip-%02d' % (6)
    print(f"Data path: {path}")

    # 确保read_all现在能正确运行
    try:
        video_path, ais_path, result_video, initial_time, camera_para = read_all(path)
    except (ValueError, IndexError) as e:
        print(f"Error initializing paths: {e}")
        exit()  # 如果路径初始化失败，直接退出

    parser = argparse.ArgumentParser(description="VesselSORT")

    # ... (参数部分保持不变)
    parser.add_argument("--video_path", type=str, default=video_path)
    parser.add_argument("--ais_path", type=str, default=ais_path)
    parser.add_argument("--result_video", type=str, default=result_video)
    # 【重要】: 这里的default应该是datetime对象，但argparse不支持，所以在main函数中直接使用
    # 我们将在下面直接传递initial_time
    parser.add_argument("--initial_time_obj", default=initial_time)  # 传递对象
    parser.add_argument("--camera_para", type=list, default=camera_para)

    # 移除废弃的参数
    # parser.add_argument("--prepare_time", type=int, default=1)
    # parser.add_argument("--anti", type=int, default=1)
    # parser.add_argument("--val", type=int, default=0)

    arg = parser.parse_args()

    # 修改argparse处理datetime对象的方式
    # argparse会将所有东西转为字符串，所以我们直接用上面读取到的initial_time对象
    arg.initial_time = initial_time

    print("\nVesselSORT")
    for p, v in vars(arg).items():
        if p != 'initial_time_obj':  # 不打印这个临时的
            print(f'\t{p}: {v}')
    print('\n')

    main(arg)