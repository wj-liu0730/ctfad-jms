import os

# here we need to include some abnormal images to the validation set instead of randomly split
# due to the data imbalance of the anomaly detection task


val_image_names = ["0e912f26-2nd_010", "1a74b680-010", "0af6c54d-2nd_021", "1f23ced0-006", "49b3d9dc-2nd_009"]

all_dataset_path = 'dataset/weldingspot/'
dataset_name_list = ["dataset_whole", "dataset_split", "dataset_anomaly"]

for dataset_name in dataset_name_list:
    # split dataset for dataset_whole
    dataset_path = os.path.join(all_dataset_path, dataset_name)
    data_name_list = os.listdir(os.path.join(dataset_path, 'images'))
    train_list = []
    val_list = []

    for data_name in data_name_list:
        image_name = data_name.split('.')[0]
        if dataset_name == "dataset_whole":
            data_id = image_name
        else:
            data_id = image_name.split('_')[:-1]
            data_id = '_'.join(data_id)

        if data_id in val_image_names:
            val_list.append(image_name)
        else:
            train_list.append(image_name)

    train_txt = open(os.path.join(dataset_path, 'train.txt'), 'w')
    for image_name in train_list:
        train_txt.write(image_name + '\n')
    train_txt.close()
    val_txt = open(os.path.join(dataset_path, 'val.txt'), 'w')
    for image_name in val_list:
        val_txt.write(image_name + '\n')
    val_txt.close()
