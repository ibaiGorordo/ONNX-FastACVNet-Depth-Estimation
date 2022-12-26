import cv2
import numpy as np
from fast_acvnet import FastACVNet, CameraConfig

# Initialize videos
left_cap = cv2.VideoCapture(
    "https://ingmec.ual.es/~jlblanco/malaga-urban-dataset/videos/malaga-urban-dataset_STEREO_LEFT.avi")
right_cap = cv2.VideoCapture(
    "https://ingmec.ual.es/~jlblanco/malaga-urban-dataset/videos/malaga-urban-dataset_STEREO_RIGHT.avi")
start_time = 66 * 60  # skip first {start_time} seconds
left_cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * left_cap.get(cv2.CAP_PROP_FPS))
right_cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * right_cap.get(cv2.CAP_PROP_FPS))

# Camera parameters for calibration
# TODO: Fix with the values with the correct configuration for YOUR CAMERA
# Ref: https://ingmec.ual.es/~jlblanco/malaga-urban-dataset/calibration/camera_params_raw_1024x768.txt
K_left = np.array([[837.619011, 0, 522.434637], [0, 839.808333, 402.367400], [0, 0, 1]])
D_left = np.array([-3.636834e-001, 1.766205e-001, 0.000000e+000, 0.000000e+000, 0.000000e+000])
K_right = np.array([[835.542079, 0, 511.127987], [0, 837.180798, 388.337888], [0, 0, 1]])
D_right = np.array([-3.508059e-001, 1.538358e-001, 0.000000e+000, 0.000000e+000, 0.000000e+000])
R = np.array([[9.9997625494747e-001, -6.3729476131001e-003, -2.6220373684323e-003],
              [6.3750339453031e-003, 9.9997936870410e-001, -7.8810427338438e-004],
              [2.6169607251553e-003, -8.0480113703670e-004, 9.9999625189882e-001]])
T = np.array([[1.194711e-001], [-3.144088e-004], [1.423872e-004]])

# Initialize Rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_right, D_right, K_left, D_left, (1024, 768), R, T)
map11, map12 = cv2.initUndistortRectifyMap(K_left, D_left, R2, P2, (1024, 768), cv2.CV_32FC1)
map21, map22 = cv2.initUndistortRectifyMap(K_right, D_right, R1, P1, (1024, 768), cv2.CV_32FC1)

# Camera options: baseline (m), focal length (pixel) and max distance
input_shape = (480, 640)  # (height, width)
camera_config = CameraConfig(T[0], K_left[0, 0] * input_shape[1] / 1024)
max_distance = 35

# Initialize model
model_path = f'models/fast_acvnet_plus_generalization_opset16_{input_shape[0]}x{input_shape[1]}.onnx'
depth_estimator = FastACVNet(model_path, camera_config=camera_config, max_dist=max_distance)

cv2.namedWindow("Estimated depth", cv2.WINDOW_NORMAL)
while left_cap.isOpened() and right_cap.isOpened():

    # Read frame from the video
    ret, left_frame = left_cap.read()
    ret, right_frame = right_cap.read()

    if not ret:
        break

    # Rectify the images
    left_frame = cv2.remap(left_frame, map11, map12, cv2.INTER_LINEAR)
    right_frame = cv2.remap(right_frame, map21, map22, cv2.INTER_LINEAR)

    # Estimate the depth
    disparity_map = depth_estimator(left_frame, right_frame)
    color_depth = depth_estimator.draw_depth()
    combined_image = cv2.addWeighted(left_frame, 0.6, color_depth, 0.4, 0)

    cv2.imshow("Estimated depth", combined_image)

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break
