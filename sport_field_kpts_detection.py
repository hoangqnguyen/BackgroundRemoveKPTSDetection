import os
import time

import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from model.attention_effv2sunet import Unet
from utils.inference import get_final_preds, get_final_preds_torch
from utils.utils import to_torch, normalize
from video_dataset import get_dataloader

def calculate_homography(
    kpts_model,
    dataloader,
    device,
    distance_threshold,
    num_keypoints,
    threshold,
    input_size,
    hm_size=(384, 384),
):
    frame_ids = []
    org_sizes = []
    num_kpts_all = []
    # predictions = []
    time_hm = []
    time_filter = []
    preds, maxvals = [], []

    print("Predicting keypoints from frames ...")

    kpts_model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            frame_id_batch = batch["frame_id"]
            frames = batch["frame"].to(device)
            org_size_batch = batch["org_size"]

            frame_ids.extend(frame_id_batch.to(int).tolist())
            org_sizes.extend(org_size_batch.to(int).tolist())

            # Prediction
            predictions_batch = kpts_model(frames)
            preds_batch, maxvals_batch = get_final_preds_torch(predictions_batch)

            preds.extend(np.split(preds_batch, preds_batch.shape[0], axis=0))
            maxvals.extend(np.split(maxvals_batch, maxvals_batch.shape[0], axis=0))
            # predictions.extend(predictions_batch.split(1, dim=0))

    print("Post-processing model outputs..")
    
    # for pre, org_size in tqdm(
    #     zip(predictions, org_sizes), total=len(predictions)
    # ):
    for idx in tqdm(range(len(preds))):
        # t_hm = time.time()
        # inference file
        # preds, maxvals = get_final_preds(prediction.numpy())
        # time_hm.append(time.time() - t_hm)

        # t_filter = time.time()
        filtered_keypoints = []
        for i in range(num_keypoints):
            if maxvals[idx][0, i, :] >= threshold:
                pred_kpts = preds[idx][0, i, :]
                x = np.rint(pred_kpts[0] * org_sizes[idx][0] / hm_size[0]).astype(np.int32)
                y = np.rint(pred_kpts[1] * org_sizes[idx][1] / hm_size[1]).astype(np.int32)

                keypoint = (x, y)
                # Compare the distance between the current keypoint and all other keypoints
                distances = [
                    np.linalg.norm(np.array(keypoint) - np.array(kp))
                    for kp in filtered_keypoints
                ]
                if len(filtered_keypoints) == 0 or min(distances) > distance_threshold:
                    filtered_keypoints.append(keypoint)
                else:
                    filtered_keypoints.append((0.0, 0.0))
            else:
                # pts.append((0, 0))
                filtered_keypoints.append((0.0, 0.0))


        pts = np.array(filtered_keypoints).reshape(-1, 2)

        pts_sel, template_sel = [], []
        for kp_idx, kp in enumerate(pts):
            if (
                int(kp[0]) != 0
                and int(kp[1]) != 0
                and (0 <= int(kp[0]) < org_sizes[idx][0])
                and (0 <= int(kp[1]) < org_sizes[idx][1])
            ):
                x = int(kp[0])
                y = int(kp[1])
                pts_sel.append((x, y))
        # time_filter.append(time.time() - t_filter)

        pts_sel = np.array(pts_sel)
        num_kpts = len(pts_sel)
        num_kpts_all.append(num_kpts)

    # print("Average time for heatmap: ", np.mean(time_hm))
    # print("Average time for filtering: ", np.mean(time_filter))
    return num_kpts_all, frame_ids


def main(
    video_path,
    output_dir,
    kpts_checkpoint_path="./checkpoints/volleyball_best_latest.pth",
    threshold=0.8,
    distance_threshold=30,
    num_keypoints=24,
    input_size=(384, 384),
    batch_size=32,
    num_workers=8,
    device="cuda",
):
    start_time = time.time()

    video_fn = os.path.basename(video_path)[:-4]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "{}.csv".format(video_fn))

    print("Loading model...")
    kpts_model = Unet(out_ch=num_keypoints + 1)
    model_dict = torch.load(kpts_checkpoint_path)
    kpts_model.load_state_dict(model_dict, strict=True)
    kpts_model = kpts_model.to(device)
    kpts_model.eval()

    segment_dict = {}
    frame_ids, frames, nums = [], [], []

    # print("Loading frames...")
    print("Detecting keypoints...")
    dataloader = get_dataloader(
        video_path, batch_size=batch_size, resize=input_size, num_workers=num_workers
    )
    nums, frame_ids = calculate_homography(
        kpts_model,
        dataloader,
        device,
        distance_threshold,
        num_keypoints,
        threshold,
        input_size=input_size,
    )

    end_time = time.time()
    print("Saving results...")

    segment_dict["frame"] = frame_ids
    segment_dict["num_kpts"] = nums

    df = pd.DataFrame.from_dict(segment_dict, orient="index").transpose()
    df.to_csv(out_path, header=True, index=False)

    duration = end_time - start_time
    print("Duration: {}".format(duration))
    print("==*==*" * 15)


if __name__ == "__main__":
    dataset_dir = "./"
    videos_dir = os.path.join(dataset_dir, "videos")
    output_dir = os.path.join(dataset_dir, "SFR_batch", "segmentations")
    vdo_path = os.path.join(videos_dir, "v_2324_223_s2_short.mp4")

    configs = {
        "video_path": vdo_path,
        "output_dir": output_dir,
        "kpts_checkpoint_path": "./checkpoints/volleyball_best_latest.pth",
        "threshold": 0.8,
        "distance_threshold": 30,
        "num_keypoints": 24,
        "input_size": (384, 384),
        "batch_size": 32,
        "num_workers": 8,
        "device": "cuda",
    }

    # season_id = ['2324']
    # game_id = ['246', '247', '248', '249', '250']
    # set_id = ['s2', 's3']

    # for ssid in season_id:
    #     for gid in game_id:
    #         for sid in set_id:
    #             vdo_path = os.path.join(videos_dir, 'v_{}_{}_{}.mp4'.format(ssid, gid, sid))

    main(**configs)
