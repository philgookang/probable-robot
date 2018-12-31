import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class MusinsaDataset(data.Dataset):
    """ Musinsa Custom Databset compatible with torch.untils.data.DataLoader """

    def __init__(self, root, idx_list, image_list, name_list, name_title):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            idx_list: list of image database idx
            image_list: list of image file name
            name_list: list of product name
            name_title:  a list of the different types of names
        """
        self.root = root
        self.idx_list   = idx_list
        self.image_list = image_list
        self.name_list  = name_list
        self.name_title = name_title


    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.image_list)


    def __getitem__(self, index):
        """ Retrieve an item from image list """

        # get image idx
        idx = self.idx_list[index]

        ## -------------

        # get image
        filename = self.image_list[index]

        # load product image
        image_item = Image.open(os.path.join(self.root, filename)).convert('RGB')

        # reshape to match pytorch
        np_image_item = np.asarray(image_item)
        np_image_item = np_image_item.reshape((3, 1500, 1800))
        np_image_item = np_image_item.astype(np.float32)

        # convert to tensor image
        image_item = torch.from_numpy(np_image_item)

        ## -------------

        # get name
        name = self.name_list[index]

        #
        lb = LabelEncoder()

        # convert title to label
        lb.fit(self.name_title)

        # get matrix
        name_matrix = lb.transform([name])

        # convert to tensor target class
        target = torch.from_numpy(name_matrix)

        # return (idx, image_item, target)
        return image_item, target
