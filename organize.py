import os
import sys
import json
#a = os.path.abspath(os.path.join(os.path.dirname(__file__), '../probable-robot-db'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../probable-robot-db')) )
from ProductBrandsM import ProductBrandsM

class Organize:

    def __init__(self):
        self.products = []

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

                    # add product to products
                    self.products.append(product)

                if pos > 30:
                    break


    def load_brand(self):
        brands = []

        for product in self.products:
            brand = self._parse_brand(product['brand'])

            # add brand to list
            brands.append(brand)

        # remove duplicate from list
        brands = list(set(brands))
        print(brands)


    def _parse_brand(self, brand_ori):

        brand_ori = brand_ori.lower()
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

    # insert all brands into the database first
    org.load_brand()
