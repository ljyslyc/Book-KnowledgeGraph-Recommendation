# -*- coding: utf-8 -*-
__author__ = "yanhe@chinasofti.com"

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

# 导入time模块
import time
# 导入json模块
import json
# 导入codecs模块
import codecs
# 导入os模块
import os

# DoubanbookPipeline类，继承Object类
# 将获取到的信息导出至json文本文件
class DoubanbookPipeline(object):

    # process_item()函数，处理每一个采集到的电影数据
    def process_item(self, item, spider):
        print("--> JSON: write to json file...")   
        # 创建输出文件夹名称
        folder_name = 'output'
        # 获取系统当前时间
        now = time.strftime('%Y-%m-%d', time.localtime())
        # 设置保存文件名称
        fileName = 'doubanbooktop250_' + now + '.json'
        # 在当前工程目录下创建文件并取得关联
        try:
            with codecs.open(folder_name + '/' + fileName, 'a', encoding='utf-8') as fp:
                # 将读取到的一条电影数据转换成json格式
                line = json.dumps(dict(item), ensure_ascii = False) + '\n'
                # 向文件写入
                fp.write(line)
        except IOError as err:
            # str()将报错对象转为字符串
             raise ("File Error: " + str(err))
        finally:
            #关闭文件链接
            fp.close()
        # 返回item
        return item