import os
import shutil
from glob import glob
import pandas as pd

def copy_and_save_segments(img_dir, segment_path, save_dir):
    df = pd.read_csv(segment_path)
    number_kpts = df['num_kpts'].tolist()

    images_path_list = glob(os.path.join(img_dir, '*.jpg'))
    images_path_list = sorted(images_path_list)

    segment = 0
    for idx, img_path in enumerate(images_path_list):
        if number_kpts[idx] >= 10:
            fn = os.path.basename(img_path)
            out_dir = os.path.join(save_dir, 'segment_{:06d}'.format(segment))
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            out_path = os.path.join(out_dir, fn)
            shutil.move(img_path, out_path)
        else:
            segment += 1
            continue

if __name__ == '__main__':
    working_dir = '/media/ankhzaya/SSD_2TB/KOVO_Dataset'
    images_dir = os.path.join(working_dir, 'all_images')
    segments_dir = os.path.join(working_dir, 'SFR', 'segmentations')

    season_id = ['2324']
    game_id = ['246']
    set_id = ['s2', 's3']

    for ssid in season_id:
        for gid in game_id:
            for sid in set_id:
                img_dir = os.path.join(images_dir, ssid, gid, sid)
                segments_path = os.path.join(segments_dir, 'v_{}_{}_{}.csv'.format(ssid, gid, sid))

                save_dir = os.path.join(working_dir, 'rally_images', ssid, gid, sid)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                copy_and_save_segments(img_dir, segments_path, save_dir)