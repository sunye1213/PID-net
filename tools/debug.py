# 判断cuda是否可以使用
import torch

def check_cuda():
    if torch.cuda.is_available():
        print('cuda is available')
    else:
        print('cuda is not available')

check_cuda()