from ultralytics import YOLO
import os
from os.path import join , basename
import numpy as np
import cv2

# 模型路径
model = YOLO('runs/segment/train1/weights/best.pt')
# 文件夹路径
source = 'data/dataset_A/test/img/'

results = model(source=source,show_labels=False,show_conf=False,boxes=False)

for result in results:

    image_shape = result.orig_shape
    img_x, img_y = image_shape[1], image_shape[0]

    image_name = basename(result.path)  # 提取图片名称
    mask_name = f"{os.path.splitext(image_name)[0]}_bin.png"  # 根据图片名称生成保存结果的名称
    pred_mask_path = join('result/mask/A/', mask_name)

    # 检测到车道时：
    if result.masks is not None and len(result.masks) > 0:
        masks_data = result.masks.data
        # mask_all = np.zeros((352,640))
        mask_all = np.zeros((672,1280))
        for index, mask in enumerate(masks_data):
            mask = np.array(mask.cpu().numpy() * 255)
            mask_all = mask_all + mask
        mask_all = cv2.resize(mask_all,(img_x,img_y))
        cv2.imwrite(pred_mask_path, mask_all)
    # 检测不到车道时：
    else:
        width , height = img_x , img_y
        black_image = np.zeros((height , width , 3) , dtype=np.uint8)
        # 保存全黑的图像为PNG文件
        cv2.imwrite(pred_mask_path , black_image)

    pred_img_path = join('result/img/', mask_name)
    result.save(pred_img_path)                 # 保存图片