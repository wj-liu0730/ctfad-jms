import os
import cv2

jpeg_dir = "data/weldingspot/dataset_whole/images_jpg"
save_folder = "data/weldingspot/dataset_whole/images"
image_name = os.listdir(jpeg_dir)

for img in image_name:
    print(img)
    image = cv2.imread(os.path.join(jpeg_dir, img))
    cv2.imwrite(os.path.join(save_folder, img.replace('.jpg', '.png')), image)