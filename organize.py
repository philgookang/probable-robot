import os
import json
import re
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
        malls = ['musinsa']

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

                    ## --------------------

                    # rename category
                    if 'cate' in product:
                        product['cat'] = product['cate']

                    # rename name
                    if 'en_name' in product:
                        product['tit'] = product['en_name']

                    # rename description
                    if 'dsc' in product:
                        product['desc'] = product['dsc']

                    ## --------------------

                    # flip category order
                    if 'cat' in product and isinstance(product['cat'], list):
                        product['cat'] = product['cat'][::-1]

                    # add product to products
                    self.products.append(product)

                #if pos > 300:
            #        break


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
        # for each level
        temp_categories = { 0 : { }, 1 : { }, 2 : { }, 3 : { }, 4 : { }, 5 : { } }

        # filter products that do not have category
        filter_products_with_category = list(filter(lambda product : 'cat' in product, self.products))

        # loop through each product and add proper category
        for product in filter_products_with_category:

            # check if category is a list format or a string format
            # check to list format for easy processing
            categories = product['cat'] if isinstance(product['cat'], list) else list(product['cat'])

            # save category to dictionary
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


    def load_products(self):

        for product in self.products:

            pm = ProductsM()
            pm.mall_idx     = 1 # 무신사 고정
            if 'brand' in product:
                brand = self.brands[ product['brand'].lower() ]
                pm.brand_idx = brand['idx']

                pb = ProductBrandsM()
                pb.idx = brand['idx']
                pb.increaseCounter()
            else:
                pm.brand_idx = 0

            pm.name         = product['tit']
            pm.name_ori     = product['tit']
            pm.second_name  = product['sub_tit']    if 'sub_tit'    in product else ''
            pm.name_ko      = product['ko_name']    if 'ko_name'    in product else ''
            pm.description  = product['desc']       if 'desc'       in product else ''
            pm.price        = re.sub(r"\D", "", product['pri'])

            # check if price is empty
            if pm.price == '':
                # skip price
                pm.price = -999

            pm.product_id   = product['pro_id']
            pm.mp_id        = product['id']


            ## ---------------------------

            # make name variable
            name = pm.name

            # remove product number : use product id to remove
            # example
            # 1. 리버티 베이직 슬립온 SVS7205LB71
            # 2. 모던지퍼 하프부츠 DL10133 DA:EL
            # exception
            # 1. FS7WO34WBK
            pattern = re.compile(r"\((.*?)\)")
            result = pattern.search(pm.product_id)
            if result:
                product_id = result.group(1)
                name = name.replace(product_id, "")

            # change text to english now
            name = name.lower()

            # color processing
            # replace [] & () color text
            colors = ["black", "beige", "mint", "white", "red", "brown", "navy", "grey", "yellow", "light blue", "pink", "orange", "green", "blue"]
            colors_translate = {
                "yellow"        : "옐로우",
                "black"         : "블랙",
                "beige"         : "베이지",
                "mint"          : "민트",
                "white"         : "화이트",
                "red"           : "빨간",
                "brown"         : "갈색",
                "navy"          : "네이비",
                "grey"          : "그레이",
                "pink"          : "핑크",
                "light blue"    : "라이브 블루",
                "orange"        : "오랜지",
                "blue"          : "블루",
                "green"         : "초록",
                "meta silver"   : "메타 실버"
            }
            for c in colors:
                name = name.replace("({0})".format(c), " " + c) # (  )
                name = name.replace("[{0}]".format(c), " " + c) # [ ]
                name = name.replace("")

            # remove [****]
            # example
            # 1. [18SS 신상]
            # 2. [해외]
            pattern = re.compile(r"\[(.*?)\]")
            name = pattern.sub("", name)

            # remove (****)
            # exmaple
            # 1. (943344-008)
            # 2. (그레이)
            pattern = re.compile(r"\((.*?)\)")
            name = pattern.sub("", name)

            # convert _ (underbar)
            # example
            # 1. REKKEN Pumps_ROCHE RK216
            # 2. Lily One Piece_Black
            name = name.replace("_", " ")

            # convert - (dash)
            # example
            name = name.replace("-", " ")

            # remove * symbol
            # example
            name = name.replace("*", "")

            # remove seasons
            # example
            # 1. 18 S/S
            name = name.replace("18 s/s", "")

            # remove whitespaces
            name = name.rstrip()

            # save processed name
            pm.name = name

            ## --------------------------

            ## exclustion

            # price not found
            pm.ml_exclude_price = 1 if pm.price == -999 else 0

            # check if its korean or not
            if all(ord(char) < 128 for char in name):
                pm.ml_exclude_lang = 1
            else:
                pm.ml_exclude_lang = 0

            ## --------------------------

            # check if category is a list format or a string format
            # check to list format for easy processing
            if 'cat' not in product:
                product['cat'] = list()

            # categories list
            categories = product['cat'] if isinstance(product['cat'], list) else list(product['cat'])

            pm.category_1_idx = 0
            pm.category_2_idx = 0
            pm.category_3_idx = 0
            pm.category_4_idx = 0

            for key,val in enumerate(categories):
                if key in self.categories:
                    if val in self.categories[key]:

                        # increase counter
                        pc = ProductCategoriesM()
                        pc.idx = self.categories[key][val]
                        pc.increaseCounter()

                        # set product category idx number
                        if key == 0:
                            pm.category_1_idx = self.categories[key][val]
                        elif key == 1:
                            pm.category_2_idx = self.categories[key][val]
                        elif key == 2:
                            pm.category_3_idx = self.categories[key][val]
                        elif key == 3:
                            pm.category_4_idx = self.categories[key][val]

            # after inserting to database
            # save product idx
            product_idx = pm.create()

            # now, insert image to database
            self.load_image(product, product_idx)


    def load_image(self, product, product_idx):

        for index,img in enumerate(product['img']):

            # split image path by /
            components = img.split("/")

            # retrieve filename from image path components
            filename = components[len(components)-1]

            ## ------------------------

            # remove detail_
            filename = filename.replace('detail_', '')

            # remove _big
            filename = filename.replace('_big', '')

            # remove 500
            filename = filename.replace('_500', '')

            ## ------------------------

            # create product image
            pi              = ProductImagesM()
            pi.sort_idx     = index
            pi.product_idx  = product_idx
            pi.filename     = filename
            pi.create()


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

    # load products into database
    org.load_products()
