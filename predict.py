from PIL import Image
import torch, torchvision

from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import tensorflow as tf

config_file_path = "output.yml"

weights_path = "Model_Files/model_final.pth"

model_path = "Model_Files/try7_output.h5"

model = tf.keras.models.load_model("Model_Files/model_inception_train2_80.h5")


def detect(path):

    im = cv2.imread(path)

    if im.shape[1] > 4000 or im.shape[0] > 4000:
        scale_percent = 10 # percent of original size
        width = int(im.shape[1] * scale_percent / 100)
        height = int(im.shape[0] * scale_percent / 100)
        dim = (width, height)
        im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)

    cfg = get_cfg()
    cfg.merge_from_file("output.yml")
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    cfg.DATASETS.TRAIN = ('plant_train')
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['leaf']
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    op = v.get_image()[:, :, ::-1]
    cv2.imwrite('static/op.jpg', op)
    class_res_list = []
    bboxes = outputs['instances'].pred_boxes.tensor.tolist()
    print(type(bboxes))
    if len(bboxes) == 0:
        return 0
    for index in range(len(bboxes)):
        bbox = list(map(int, bboxes[index]))
        mask = outputs["instances"].pred_masks.cpu().numpy()[index]
        bbox_h = int(math.ceil(bbox[3] - bbox[1])) #Height of the predicted bounding box
        bbox_w = int(math.ceil(bbox[2] - bbox[0])) #Width of the predicted bounding box
        temp_mask = np.zeros((bbox_h, bbox_w))         #Creating a dummy image of the size of predicted bounding box
        for x_idx in range(int(bbox[1]), int(bbox[3])):
            for y_idx in range(int(bbox[0]), int(bbox[2])):
                temp_mask[x_idx - int(bbox[1])][y_idx - int(bbox[0])] = mask[x_idx][y_idx] 
        crop_img = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        gray_three = cv2.merge([temp_mask,temp_mask,temp_mask])
        leaf = (gray_three*crop_img)
        cv2.imwrite('static/op1' + 'index' + '.jpg', leaf)
        print(type(leaf))
        #leaf  = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)
        #leaf = np.random.random_sample(leaf.shape) * 255
        leaf = leaf.astype(np.uint8)    
        leaf_pil = Image.fromarray(leaf)
        leaf_pil = leaf_pil.resize((224, 224))
        leaf_pil = np.asarray(leaf_pil, dtype = np.float32)
        leaf_pil = leaf_pil /255 #check if horrible performance
        leaf_pil = leaf_pil.reshape(-1, 224, 224, 3)
        prediction = model.predict(leaf_pil)
        prediction = np.argmax(prediction)
        class_res_list.append(prediction)
    #img_name = path.split('/')[-1].split('.')[0]
    #img_ext = path.split('/')[-1].split('.')[1]
    #op_filename =  img_name + '_op.' + img_ext
    #cv2.imwrite("D:\LY\WebApp\static\output\\" + op_filename, op)
    classes = ['Aboli', 'Aglaonema Siam', 'Anant', 'Cannon ball tree', 'Coleus', 'Croton', 'Cuphea', 'Dieffenbachia', 'Goeppertia', 'Gokarn', 'Henna', 'Indian Tulip Tree', 'Insulin', 'Jamaican Spike', 'Kindal Tree', 'Money Plant', 'Mussaenda', 'Nirgudu', 'Paanphuti', 'Papai', 'Pentas', 'Poinsettia', 'Polka', 'Ratrani', 'Saplera', 'Singoniya', 'South Indian soapnut', 'Spanish Cherry', 'Wild Henna']
    dict1 = {}
    for cls in class_res_list:
        if classes[cls] not in dict1.keys():
            dict1[classes[cls]] = 1
        else:
            dict1[classes[cls]] += 1
    dictRest = sorted(dict1.items(), key=lambda x:x[1])
    resSortedDict = dict(dictRest)
    result = ''
    for i in reversed(list(resSortedDict)):
        result = result  + i  + ', '
    return result