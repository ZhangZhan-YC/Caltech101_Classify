import pandas as pd
import os
import glob

train_dir = 'D:/DataBase/Calteck101/train'
valid_dir = 'D:/DataBase/Calteck101/valid'
test_dir = 'D:/DataBase/Calteck101/test'

train_csv_dir = 'D:/DataBase/Calteck101/train.csv'
valid_csv_dir = 'D:/DataBase/Calteck101/valid.csv'
test_csv_dir = 'D:/DataBase/Calteck101/test.csv'


def generate_csv(csv_dir, src_dir):
    img_dir_list = list()
    label_list = list()
    for category in os.listdir(src_dir):
        for img_dir in glob.glob(os.path.join(src_dir, category) + '/*.jpg'):
            label_list.append(category)
            img_dir_list.append(img_dir)
    pd.DataFrame({'img_dir': img_dir_list, 'label': label_list}).to_csv(csv_dir, index=False, encoding='utf8')


if __name__ == '__main__':
    generate_csv(train_csv_dir, train_dir)
    generate_csv(valid_csv_dir, valid_dir)
    generate_csv(test_csv_dir, test_dir)
