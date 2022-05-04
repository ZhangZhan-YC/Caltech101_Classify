# test for os.walk()
#
# import os
#
# data_dir = 'D:/DataBase/caltech-101'
# i = 0
#
# for root, dirs, _ in os.walk(data_dir):
#     print(root)
#     print(dirs)
#     print(_)
#     if i > 2:
#         break
#     i += 1

#
# read for .mat
#
# from scipy.io import loadmat
# dataMat=loadmat('D:/DataBase/caltech-101/Annotations/Airplanes_Side_2/annotation_0001.mat')
# # mat文件有很多不相关的keys，要找到目标的keys
# # print(dataMat.keys()) # 输出为dict_keys(['__header__', '__version__', '__globals__', 'data', 'label'])
# temp1 = dataMat['__header__']
# temp2 = dataMat['__version__']
# temp3 = dataMat['__globals__']
# temp4 = dataMat['box_coord']
# temp5 = dataMat['obj_contour']
# print(temp1)
# print(temp2)
# print(temp3)
# print(temp4)
# print(temp5)
# print(temp5.shape)

#
# test for dataloader
#
# from Caltech101_Dataset import Caltech101Dataset
# from torch.utils.data import DataLoader
# from torchvision.transforms import transforms
#
# train_csv_dir = 'D:/DataBase/Calteck101/train.csv'
# valid_csv_dir = 'D:/DataBase/Calteck101/valid.csv'
# test_csv_dir = 'D:/DataBase/Calteck101/test.csv'
#
# transform = transforms.Compose([
#     transforms.Resize([224, 224]),
#     transforms.ToTensor()
# ])
#
# train_dataset = Caltech101Dataset(train_csv_dir, transform)
#
# train_loader = DataLoader(dataset=train_dataset, batch_size=32)
#
# rest of the images in the last step is abandoned!!! adjust the parameters!!!
# for epoch in range(10):
#     for step, data in enumerate(train_loader):
#         inputs, labels = data
#         print("epoch:", epoch, "step:", step)
#         print("data shape:", inputs.shape, labels.shape)




