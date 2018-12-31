import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from MusinsaDataset import *
from config import *

class Pretrain(nn.Module):

    def __init__(self, **kwargs):
        super(Pretrain, self).__init__()

        self.batch_size             = kwargs['batch_size']
        self.padding                = kwargs['padding']
        self.stride                 = kwargs['stride']
        self.num_workers            = kwargs['num_workers']
        self.number_of_images       = kwargs['number_of_images']
        self.limit_loading_count    = kwargs['limit_loading_count']  # number to load each time
        self.image_width            = kwargs['image_width']
        self.image_height           = kwargs['image_height']

        # get data
        self.get_train_data()

        print("****************************")
        print("Data Params: ")
        print("number of images  :    ", self.number_of_images)
        print("# of img to load  :    ", self.limit_loading_count)
        print("total data        :    ", len(self.train_loader))
        print("number of classes :    ", self.num_classes)
        print()
        print("Model Params: ")
        print("batch_size        :    ", self.batch_size)
        print("padding           :    ", self.padding)
        print("stride            :    ", self.stride)
        print("num_workers       :    ", self.num_workers)
        print("****************************")

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5))
        self.fc = nn.Linear(64*60*72, self.num_classes)

    def get_train_data(self):

        # holds complete list of images
        product_image_list = list()

        # loop load to prevent memory issue
        for i in range(1000):

            # offset from begining
            offset = (i*self.limit_loading_count)

            # check if we have loaded enough
            if offset > self.number_of_images:
                break

            # retreive data from database
            pi = ProductImagesM()
            pi.width_pixel = 1500
            pi.height_pixel = 1800
            tmp_list = pi.getMLList( offset = offset, limit = self.limit_loading_count )

            # if no more to load
            # break loop
            if (len(tmp_list) == 0):
                break

            # add to list
            product_image_list.extend(tmp_list)

        image_directory     = "/mnt/ssd3/probable-robot/ui/public/images/"
        idx_list            = list(map(lambda x : x["idx"],             product_image_list))
        image_list          = list(map(lambda x : x["filename"],        product_image_list))
        name_list           = list(map(lambda x : x["name"],            product_image_list))
        category_list       = list(map(lambda x : x["category_name"],   product_image_list))
        category_title      = list(set(category_list))
        self.num_classes    = len(category_title)

        # setup musinsa dataset
        musinsa_dataset = MusinsaDataset(image_directory, idx_list, image_list, category_list, category_title)

        # create data loader
        self.train_loader = DataLoader(musinsa_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        num_workers=self.num_workers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
