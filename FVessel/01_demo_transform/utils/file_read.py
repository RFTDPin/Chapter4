import os
import glob
import re
from datetime import datetime, timedelta


def read_all(path):
    """
    这个函数基本保持不变，只是返回的initial_time现在是一个datetime对象。
    """
    name = path.split('/')[-1]
    video_path = (glob.glob(path + '/*.mp4') + glob.glob(path + '/*.avi'))[0]

    # 使用正则表达式从视频文件名中解析初始时间
    # 假设文件名格式为 ...-YYYY-MM-DD-HH-MM-SS.mp4
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', video_path)
    if not match:
        raise ValueError(f"Could not parse initial time from video filename: {video_path}")

    time_parts = [int(p) for p in match.groups()]
    initial_time_obj = datetime(time_parts[0], time_parts[1], time_parts[2],
                                time_parts[3], time_parts[4], time_parts[5])

    ais_path = path + '/ais'
    result_video = './result/out_%s.mp4' % (name)

    # 读取相机参数
    with open(glob.glob(path + '/*.txt')[0], "r") as f:
        camera_para_str = f.readlines()[0][1:-2]
        camera_para = list(map(float, camera_para_str.split(',')))

    return video_path, ais_path, result_video, initial_time_obj, camera_para


def time_to_key(dt_obj):
    """
    【新增函数】
    将datetime对象转换为用于AIS数据字典查询的key。
    这个key的格式必须和preload_all_ais_data中生成key的格式完全一致。
    这里我们统一使用 'YYYY_MM_DD_HH_MM_SS' 格式。
    """
    return dt_obj.strftime('%Y_%m_%d_%H_%M_%S')


def update_time(current_time_obj, ms_to_add):
    """
    【重构后的函数】
    使用datetime.timedelta来安全、准确地更新时间。
    :param current_time_obj: 当前时间的datetime对象
    :param ms_to_add: 需要增加的毫秒数
    :return: (新的datetime对象, 对应的Unix时间戳(毫秒), 用于查询的key)
    """
    new_time_obj = current_time_obj + timedelta(milliseconds=ms_to_add)

    # 转换为毫秒级Unix时间戳
    timestamp_ms = int(new_time_obj.timestamp() * 1000)

    # 生成用于查询的key (只精确到秒)
    time_key = time_to_key(new_time_obj)

    return new_time_obj, timestamp_ms, time_key


# ais_initial函数在重构后不再需要，因为我们预加载了所有数据。
# time2stamp函数的功能也被新的update_time和time_to_key所取代。
# 但如果你在别的地方还需要它，可以保留一个简化版的。

def datetime_to_info(dt_obj):
    """
    【可选的替代函数】
    替代原来的time2stamp，接收一个datetime对象。
    """
    timestamp_ms = int(dt_obj.timestamp() * 1000)
    name_str = dt_obj.strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]  # 精确到毫秒
    return timestamp_ms, name_str