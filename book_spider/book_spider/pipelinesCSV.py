import csv
import itertools
import time
 
class DoubanbookPipeline(object):

    # __init__()函数为该类的构造函数，主要实现创建一个csv文件
    def __init__(self):
        # 创建输出文件夹名称
        folder_name = 'output'
        # 获取系统当前时间
        now = time.strftime('%Y-%m-%d', time.localtime())
        # 设置保存文件名称
        fileName = 'doubanbooktop250_' + now + '.csv'
        # 最终文件路径
        self.excelPath = folder_name + '/' + fileName      
        # 创建csv
        self.csvwriter = csv.writer(open(self.excelPath, 'a+', encoding='utf-8',newline=''), delimiter=',')
        self.csvwriter.writerow(['book_name','english_name','author','img','publish','price','score','description'])
    def close_spider(self,spider):
        #关闭爬虫时顺便将文件保存退出
        self.csvwriter.close()
    def process_item(self, item, ampa):

        rows = [item['book_name'],item['english_name'],item['author'],item['img'],item['publish'],item['price'],item['score'],item['description']]
 
        self.csvwriter.writerow(rows)
 
        return item