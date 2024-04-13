# 读取 ../data/mapill文件夹，生成test.lst train.lst val.lst trainval.lst四个文件
# 生成的文件存储在../data/list/mapill/文件夹下
# 生成的文件内容格式为：图片路径 图片标签路径
# train.lst读取../data/mapill/image/train文件夹下的图片路径，与../data/mapill/label/train文件夹下面的图片路径，生成train.lst文件
# val.lst读取../data/mapill/image/val文件夹下的图片路径，与../data/mapill/label/val文件夹下面的图片路径，生成val.lst文件
# test.lst读取../data/mapill/image/test文件夹下的图片路径，生成test.lst文件
# 图片路径与图片标签路径之间用空格隔开 不同图片之间用换行符隔开
# 生成的文件内容格式为：图片路径 图片标签路径

import os

# 生成的文件内容格式为：图片路径 图片标签路径
# 图片以jpg存储，标签以png存储
def get_list(data_dir, list_dir):
    for phase in ['train', 'val', 'test']:
        with open(os.path.join(list_dir, phase + '.lst'), 'w') as f:
            for root, _, files in os.walk(os.path.join(data_dir, 'image', phase)):

                for file in files:
                    if file.endswith('.jpg'):
                        if phase != 'test':
                            # 把每一个图片的路径和标签路径 ../data/mapill/去掉，只保留后面的路径
                            # 如image/train/kMRZe3cb4J_eAWrb5NO2Ug.jpg label/train/kMRZe3cb4J_eAWrb5NO2Ug.png
                            # 而不是/image/train/kMRZe3cb4J_eAWrb5NO2Ug.jpg /label/train/kMRZe3cb4J_eAWrb5NO2Ug.png
                            img_path = os.path.join(root, file)[len(data_dir)+1:]
                            label_path = img_path.replace('image', 'label').replace('.jpg', '.png')
                            f.write(img_path + ' ' + label_path + '\n')
                        else:
                            img_path = os.path.join(root, file)[len(data_dir)+1:]
                            f.write(img_path + '\n')
                            
                            # test没有标签，所以只有图片路径
                        
if __name__ == '__main__':
    data_dir = '../data/mapill'
    list_dir = '../data/list/mapill'
    if not os.path.exists(list_dir):
        os.makedirs(list_dir)
    get_list(data_dir, list_dir)
