import os
import numpy as np
from tqdm import tqdm

label_directory = 'data/weldingspot/dataset_whole/labels'
abnormal_folder = 'data/weldingspot/dataset_anomaly/abnormal'
all_abnormal_files = os.listdir(abnormal_folder)

for ab_file in tqdm(all_abnormal_files):
    print(ab_file)
    ab_name = ab_file.split('.')[0]
    ab_name_split = ab_name.split('_')
    if len(ab_name_split) < 3:
        image_name = ab_name_split[0]
    else:
        image_name = ab_name_split[0] + '_' + ab_name_split[1]
    bbox_id = int(ab_name_split[-1])

    label_file = os.path.join(label_directory, f'{image_name}.txt')
    labels = np.loadtxt(label_file)
    labels[bbox_id][0] = 1
    text_file = open(label_file, 'w')
    for line in labels:
        for i, item in enumerate(line):
            if i == 0:
                item = int(item)
            text_file.write(f"{item} ")
        text_file.write("\n")
    text_file.close()

