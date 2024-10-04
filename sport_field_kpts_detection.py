import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from model.attention_effv2sunet import Unet
from utils.inference import get_final_preds
from utils.utils import to_torch, normalize


def calculate_homography(kpts_model, frame, device, distance_threshold, num_keypoints, threshold):
    input_size = (384, 384)
    hm_size = (384, 384)
    org_size = (frame.shape[1], frame.shape[0])
    input_image = cv2.resize(frame, (input_size[0], input_size[1]))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = to_torch(input_image).permute(2, 0, 1)
    input_image = normalize(input_image)
    input_image = input_image.expand(1, -1, -1, -1)

    input_image = input_image.to(device, non_blocking=True).float()

    # Prediction
    prediction = kpts_model(input_image)
    prediction = prediction.to(device).float()

    # inference file
    preds, maxvals = get_final_preds(prediction.clone().cpu().detach().numpy())

    filtered_keypoints = []
    for i in range(num_keypoints):
        if maxvals[0, i, :] >= threshold:
            pred_kpts = preds[0, i, :]
            x = np.rint(pred_kpts[0] * org_size[0] / hm_size[0]).astype(
                np.int32)
            y = np.rint(pred_kpts[1] * org_size[1] / hm_size[1]).astype(
                np.int32)

            keypoint = (x, y)
            # Compare the distance between the current keypoint and all other keypoints
            distances = [np.linalg.norm(np.array(keypoint) - np.array(kp)) for kp in filtered_keypoints]
            if len(filtered_keypoints) == 0 or min(distances) > distance_threshold:
                filtered_keypoints.append(keypoint)
            else:
                filtered_keypoints.append((0., 0.))
        else:
            # pts.append((0, 0))
            filtered_keypoints.append((0., 0.))

    pts = np.array(filtered_keypoints).reshape(-1, 2)

    pts_sel, template_sel = [], []
    for kp_idx, kp in enumerate(pts):
        if int(kp[0]) != 0 and int(kp[1]) != 0 and (0 <= int(kp[0]) < org_size[0]) and (
                0 <= int(kp[1]) < org_size[1]):
            x = int(kp[0])
            y = int(kp[1])
            pts_sel.append((x, y))

    pts_sel = np.array(pts_sel)
    num_kpts = len(pts_sel)

    return num_kpts


def main(path, output_folder):
    start_time = time.time()
    checkpoints_dir = './checkpoints/'

    video_fn = os.path.basename(path)[:-4]
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, '{}.csv'.format(video_fn))

    kpts_checkpoint_path = os.path.join(checkpoints_dir, 'volleyball_best_latest.pth')
    threshold = 0.8
    distance_threshold = 30
    num_keypoints = 24
    device = torch.device("cuda")

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Video path is : {}, FPS: {}".format(path, fps))

    frame_id = 0

    kpts_model = Unet(out_ch=num_keypoints + 1)
    model_dict = torch.load(kpts_checkpoint_path)
    kpts_model.load_state_dict(model_dict, strict=True)
    kpts_model = kpts_model.to(device)
    kpts_model.eval()

    segment_dict = {}
    frames, nums = [], []

    while True:
        ret_val, frame = cap.read()
        if ret_val:
            num_kpts = calculate_homography(kpts_model, frame, device, distance_threshold, num_keypoints, threshold)
            # print("Frame ID : {}, Num of KPTS: {}".format(frame_id, num_kpts))

            frames.append(frame_id)
            nums.append(num_kpts)

        else:
            break
        frame_id += 1

    end_time = time.time()

    segment_dict['frame'] = frames
    segment_dict['num_kpts'] = nums

    df = pd.DataFrame.from_dict(segment_dict, orient='index').transpose()
    df.to_csv(out_path, header=True, index=False)

    duration = end_time - start_time
    print('Duration: {}'.format(duration))
    print('==*==*' * 15)


if __name__ == "__main__":
    dataset_dir = "./"
    videos_dir = os.path.join(dataset_dir, 'videos')
    output_dir = os.path.join(dataset_dir, 'SFR_original_code', 'segmentations')

    # season_id = ['2324']
    # game_id = ['246', '247', '248', '249', '250']
    # set_id = ['s2', 's3']

    # for ssid in season_id:
    #     for gid in game_id:
    #         for sid in set_id:
    #             vdo_path = os.path.join(videos_dir, 'v_{}_{}_{}.mp4'.format(ssid, gid, sid))

    vdo_path = os.path.join(videos_dir, 'v_2324_223_s2_short.mp4')
    main(vdo_path, output_dir)