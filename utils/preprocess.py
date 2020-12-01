import cv2
import os

img_dir = "D:\\data\\train_data\\mask"
out_dir = "D:\\data\\mask"

list = os.listdir(img_dir)
print(list)
for i in range(len(list)):
    img_path = os.path.join(img_dir, list[i])
    img = cv2.imread(img_path)

    print(img.read())
