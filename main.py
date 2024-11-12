# lane_detection_pipeline/main.py
import cv2
import numpy as np
import yaml
from camera_calibration import get_camera_calibration_coefficients, undistort_image
from perspective_transform import warp_image
from binary_thresholds import (
    hls_l_select,
    hls_s_select,
    dir_threshold,
    mag_threshold,
    abs_sobel_threshold,
    lab_b_select,
)
from lane_detection import (
    find_lane_pixels,
    fit_polynomial,
    fit_poly,
    search_around_poly,
)
from curvature_offset import measure_curvature_real
from drawing_utils import drawing, draw_text


def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def main(config_file):
    # 加载配置文件
    config = load_config(config_file)

    # 提取参数
    nx = config['camera_calibration']['nx']
    ny = config['camera_calibration']['ny']
    calibration_pattern = config['camera_calibration']['calibration_pattern']
    input_video = config['output']['input_video']
    output_video = config['output']['output_video']

    src_points = np.float32(config['perspective_transform']['src_points'])
    dst_points = np.float32(config['perspective_transform']['dst_points'])

    sobel_thresh_min = config['binary_thresholds']['sobel_thresh']['thresh_min']
    sobel_thresh_max = config['binary_thresholds']['sobel_thresh']['thresh_max']

    nwindows = config['lane_detection']['nwindows']
    margin = config['lane_detection']['margin']
    minpix = config['lane_detection']['minpix']

    # Step 1: 获取畸变校正参数
    rets, mtx, dist, rvecs, tvecs = get_camera_calibration_coefficients(calibration_pattern, nx, ny)

    # 初始化视频读取和写入
    cap = cv2.VideoCapture(input_video)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read input video.")
        return

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (frame.shape[1], frame.shape[0]))

    while ret:
        # Step 2: 畸变校正
        undistorted_image = undistort_image(frame, mtx, dist)

        # Step 3: 透视变换（生成俯视图）
        warped_image, M, Minv = warp_image(undistorted_image, src_points, dst_points)

        # Step 4: 提取车道线的二值图像
        sx_binary = abs_sobel_threshold(warped_image, orient='x', thresh_min=sobel_thresh_min,
                                        thresh_max=sobel_thresh_max)
        hlsL_binary = hls_l_select(warped_image)
        labB_binary = lab_b_select(warped_image)
        combined_binary = np.zeros_like(sx_binary)
        combined_binary[(hlsL_binary == 255) | (labB_binary == 255)] = 255

        # Step 5: 使用滑窗拟合车道线
        _, left_fit, right_fit, ploty = fit_polynomial(combined_binary, nwindows=nwindows, margin=margin, minpix=minpix)

        # Step 6: 跟踪车道线
        track_result, track_left_fit, track_right_fit, ploty = search_around_poly(combined_binary, left_fit, right_fit)

        # Step 7: 计算曲率半径和偏移量
        left_curverad, right_curverad, offset = measure_curvature_real(track_left_fit, track_right_fit, ploty)
        average_curverad = (left_curverad + right_curverad) / 2

        # Step 8: 将车道线逆投影到原图
        left_fitx = track_left_fit[0] * ploty ** 2 + track_left_fit[1] * ploty + track_left_fit[2]
        right_fitx = track_right_fit[0] * ploty ** 2 + track_right_fit[1] * ploty + track_right_fit[2]
        lane_result = drawing(undistorted_image, combined_binary, left_fitx, right_fitx, ploty, Minv)
        final_result = draw_text(lane_result, average_curverad, offset)

        # 显示并写入视频
        cv2.imshow('Lane Detection', final_result)
        out.write(final_result)

        # 处理下一帧
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break
        ret, frame = cap.read()

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    config_file = "config.yaml"
    main(config_file)
