# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

# 导入time模块
import time
# 导入os模块
import os

class DoubanbookPipeline(object):
    # __init__()函数为该类的构造函数
    def __init__(self):
        # 创建文件夹名称
        self.folder_name = 'output'
        # 判断文件夹是否存在
        if not os.path.exists(self.folder_name):
            # 若不存在则创建文件夹output
            os.mkdir(self.folder_name)

    # process_item()函数，处理每一个采集到的电影数据
    def process_item(self, item, spider):
        print("--> TXT: write to text file...")     
        # 获取系统当前时间
        current_date = time.strftime('%Y-%m-%d', time.localtime())
        # 设置保存文件名称
        file_name = 'doubanbooktop250_' + current_date + '.txt'
        # 在当前工程目录下创建文件并取得关联
        try:
            with open(self.folder_name + '/' + file_name, 'a', encoding='utf-8') as fp:
                # 写入相关数据
                fp.write('>>>\n')
                fp.write('book_name:' + item['book_name'] + '\n')
                fp.write('author:' + item['author'] + '\n')
                fp.write('img:' +item['img'] + '\n')
                fp.write('english_name:' + item['english_name'] + '\n')
                fp.write('publish:' + item['publish'] + '\n')
                fp.write('price:' + item['price'] + '\n')
                fp.write('score:' + item['score'] + '\n')
                fp.write('description:' + item['description'] + '\n')
        except IOError as err:
            # str()将报错对象转为字符串
            raise ("File Error: " + str(err))
        finally:
            # 关闭文件链接
            fp.close()
        # 返回item
        return item
