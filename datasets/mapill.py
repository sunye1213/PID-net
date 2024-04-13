# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class mapill(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=66,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=2048, 
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.409, 0.451, 0.460], 
                 std=[0.253, 0.271, 0.301],
                 bd_dilate_size=4):

        super(mapill, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()

        self.label_mapping = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,
                              9:9,10:10,11:11,12:12,13:13,14:14,15:15,
                              16:16,17:17,18:18,19:19,20:20,21:21,22:22,
                              23:23,24:24,25:25,26:26,27:27,28:28,29:29,
                              30:30,31:31,32:32,33:33,34:34,35:35,36:36,
                              37:37,38:38,39:39,40:40,41:41,42:42,43:43,
                              44:44,45:45,46:46,47:47,48:48,49:49,50:50,
                              51:51,52:52,53:53,54:54,55:55,56:56,57:57,
                              58:58,59:59,60:60,61:61,62:62,63:63,64:64,
                              65 : 65}
        
        self.class_weights = torch.FloatTensor([1.6091721, 1.3610879, 0.8639482, 
                                                0.84011775, 0.9280921, 0.8928364, 
                                                0.86546034, 0.8897578, 0.9392268, 
                                                0.9771725, 0.94744116, 0.92173964, 
                                                0.9342955, 0.7394958, 0.9430677, 
                                                0.8028108, 0.8699503, 0.7510921, 
                                                0.98046845, 0.9207333, 1.0427694, 
                                                1.0583928, 1.4576918, 0.8625974, 
                                                0.8416053, 0.95416266, 1.2083519, 
                                                0.7237048, 0.9185221, 0.8544961, 
                                                0.74604553, 1.1215383, 1.0084294, 
                                                1.2019988, 1.1978514, 0.88181716, 
                                                1.0624793, 1.2822378, 1.1541036,
                                                1.0376477, 1.1801611, 1.0116676, 
                                                1.111973, 1.3224108, 1.0139482, 
                                                0.86625487, 0.98120534, 0.898789, 
                                                0.9383327, 1.001235, 0.8966613, 
                                                1.0478702, 0.9914402, 1.1520083, 
                                                0.9123485, 0.8061552, 1.2224605, 
                                                1.0013226, 1.0584896, 1.0432657, 
                                                1.2688596, 0.9102393, 1.1819572, 
                                                0.94149774, 0.8169047, 0.82813627]).cuda()
        self.bd_dilate_size = bd_dilate_size
            
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'mapill',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root,'mapill',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
