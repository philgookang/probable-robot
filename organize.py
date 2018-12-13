import os
import json
from multiprocessing import Pool
from setting import *

class Organize:

    def __init__(self):
        self.products = []
        self.brands = {}
        self.categories = {}
        self.colors = {}

    def get_products(self):

        path = '/mnt/ssd3/intellisys/data/'
        malls = ['musinsa', 'sisun', 'stylenanda']

        # loop through each malls
        for mall in malls:

            # create path to data file
            path_to_mall_data = path + mall + "/data/"

            # get list of json files
            files = os.listdir(path_to_mall_data)

            # load each json file
            for pos,jfile in enumerate(files):

                # open json file
                with open(path_to_mall_data+jfile) as f:

                    # load product
                    product = json.load(f, encoding='utf-8')

                    # flip category order
                    if 'cat' in product and isinstance(product['cat'], list):
                        product['cat'] = product['cat'][::-1]

                    # add product to products
                    self.products.append(product)

                if pos > 3:
                    break


    def load_brands(self):

        # filter products that do not have brand
        filter_products_with_brand = list(filter(lambda product : 'brand' in product, self.products))

        # map through each products in list
        brands = list(map(self._parse_brand, filter_products_with_brand))

        # remove duplicate from list
        brands = list(set(brands))

        for brand in brands:

            pb = ProductBrandsM()
            pb.name = brand[0]
            pb.name_ko = brand[1]
            pb.name_ori = brand[2]
            idx = pb.create()

            self.brands[brand[2]] = {
                "idx" : idx,
                "name" : brand[0],
                "name_ko" : brand[1],
                "name_ori" : brand[2]
            }


    def load_categories(self):

        # temp category list
        temp_categories = {
            # for each level
            0 : { },
            1 : { },
            2 : { },
            3 : { },
            4 : { },
            5 : { }
        }

        # filter products that do not have category
        filter_products_with_category = list(filter(lambda product : 'cat' in product, self.products))

        # loop through each product and add proper category
        for product in filter_products_with_category:

            # check if category is a list format or a string format
            # check to list format for easy processing
            categories = product['cat'] if isinstance(product['cat'], list) else list(product['cat'])

            for index,category in enumerate(categories):
                temp_categories[index][category] = 0

        for index in temp_categories:
            for category in temp_categories[index]:

                pc = ProductCategoriesM()
                pc.name = category
                pc.level = index
                temp_categories[index][category] = pc.create()

        # set global category
        self.categories = temp_categories


    def load_colors(self):

        # filter products that do not have category
        filter_products_with_color = list(filter(lambda product : 'color' in product, self.products))

        # loop through each product and add proper category
        for product in filter_products_with_color:

            # add color
            self.colors[product['color'].lower()] = 0

        # loop through color list
        # add to database
        for color in self.colors:
            pc = ProductColorsM()
            pc.name = color
            self.colors[color] = pc.create()


    def _parse_brand(self, product):

        brand_ori = product['brand'].lower()
        brand_english = ""
        brand_korean = ""
        is_ko_name = False

        for c in brand_ori:
            if c == '(':
                is_ko_name = True
                continue
            elif c == ')':
                continue
            elif ord(c) < 128 and not is_ko_name:
                brand_english += c
            else:
                brand_korean += c

        return (brand_english.strip(), brand_korean, brand_ori)


if __name__ == "__main__":

    # organize class
    org = Organize()

    # load all product information
    org.get_products()

    # insert all brands into the database first
    org.load_brands()

    # insert all categories into the database
    org.load_categories()

    # insert all colors into the database
    org.load_colors()
