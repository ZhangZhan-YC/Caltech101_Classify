import os
import glob
import shutil
import random
import tqdm

# directory tree:
# D:/Database
#       /caltech-101
#               /101_ObjectCategories
#                       /accordion
#                       ...other ordinary data
#       /Calteck101
#               /train
#                       /0
#                       ...other train data after splitting
#               /valid
#                       /0
#                       ...other valid data after splitting
#               /test
#                       /0
#                       ...other test data after splitting

data_dir = 'D:/DataBase/Calteck101/'
train_dir = 'D:/DataBase/Calteck101/train'
valid_dir = 'D:/DataBase/Calteck101/valid'
test_dir = 'D:/DataBase/Calteck101/test'


def check_file():
    dir_list = [data_dir, train_dir, valid_dir, test_dir]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)


def split_data():
    old_data_dir = 'D:/DataBase/caltech-101/101_ObjectCategories/'
    categories = os.listdir(old_data_dir)
    label_list = list()
    for category in tqdm.tqdm(categories):
        # the index of the fold as the label
        label = categories.index(category)
        label_list.append(label)
        category_dir = os.path.join(old_data_dir, category)
        # use glob.glob() to match the .jpg
        img_category = glob.glob(category_dir + '/*.jpg')
        # shuffle the sequence of the images
        random.shuffle(img_category)
        # split the directory of the images to train,valid,test
        train_size = int(0.8 * len(img_category))
        valid_size = int(0.1 * len(img_category))
        test_size = int(0.1 * len(img_category))
        img_train = img_category[:train_size]
        img_valid = img_category[train_size:train_size + valid_size]
        img_test = img_category[train_size + valid_size:]
        out_train_dir = os.path.join(train_dir, str(label))
        out_valid_dir = os.path.join(valid_dir, str(label))
        out_test_dir = os.path.join(test_dir, str(label))
        out_dir = [out_train_dir, out_valid_dir, out_test_dir]
        img_dir = [img_train, img_valid, img_test]
        for i in range(3):
            if not os.path.exists(out_dir[i]):
                os.mkdir(out_dir[i])
            for img in img_dir[i]:
                shutil.copy(img, os.path.join(out_dir[i], os.path.split(img)[-1]))


if __name__ == '__main__':
    check_file()
    split_data()
