# -*- coding: utf-8 -*-
__author__ = "yanhe@chinasofti.com"

# 导入urllib模块
import urllib
# 导入os模块
import os

# DoubanbookPipeline类，继承Object类
# 将获取到的图片下载到output/images文件夹下
class DoubanbookPipeline(object):

    # __init__()函数为该类的构造函数
    def __init__(self):
        # 创建文件夹名称
        self.folder_name = 'output/images'
        # 判断文件夹是否存在
        if not os.path.exists(self.folder_name):
            # 若不存在则创建文件夹output
            os.mkdir(self.folder_name)

    # process_item()函数，处理每一个采集到的电影数据
    def process_item(self, item, spider):
        print("--> 图片采集: download book images to local disk...")    
        # 获取电影详情链接地址
        book_pic = item['img']
        # 拆分字符串，并获取最后一个元素作为图片名称
        image_name = item['book_name'] + '.jpg'
        try:
            # 下载图片到指定的文件夹中
            urllib.request.urlretrieve(item['img'], self.folder_name +'/%s' %image_name)
        except Exception as e:
            # 下载报错
            raise ("Download Exception: " + str(e))
        # 返回item
        return item