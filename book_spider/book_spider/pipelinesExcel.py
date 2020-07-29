# -*- coding: utf-8 -*-
# 导入time模块
import time
# 导入xlwt模块
import xlwt
# 导入xlrd模块
import xlrd
# 导入xlutils中的copy
from xlutils.copy import copy

# DoubanbookPipeline类，继承Object类
# 将获取到的信息导出至excel文本文件
class DoubanbookPipeline(object):

    # __init__()函数为该类的构造函数，主要实现创建一个excel表格
    def __init__(self):
        # 创建输出文件夹名称
        folder_name = 'output'
        # 获取系统当前时间
        now = time.strftime('%Y-%m-%d', time.localtime())
        # 设置保存文件名称
        fileName = 'doubanbooktop250_' + now + '.xls'
        # 最终文件路径
        self.excelPath = folder_name + '/' + fileName      
        # 创建excel工作簿workbook
        self.workbook = xlwt.Workbook(encoding = 'UTF-8')
        # 创建工作簿中的sheet页
        self.sheet = self.workbook.add_sheet(u'豆瓣书籍数据')
        # 设置excel标题栏内容
        headers = ['book_name','english_name','author','img','publish','price','score','description']
        # 设置excel标题栏字体为黑色加粗
        headstyle = xlwt.easyxf('font: color-index black, bold on')        
        # 循环写入标题内容
        for colIndex in range(0,len(headers)):
            # 按照字体格式样式写入标题
            self.sheet.write(0,colIndex,headers[colIndex], headstyle)
        # 保存创建好的excel文件
        self.workbook.save(self.excelPath)
        # 设置excel行全局循环变量
        self.rowIndex = 1

    

    # process_item()函数，处理每一个采集到的电影数据并追加到已经创建好的excel文件中
    def process_item(self, item, spider):
        print("--> Excel: write to excel file...")
        # 打开已经存在的excel文件，并保留原有格式
        oldWb = xlrd.open_workbook(self.excelPath, formatting_info=True)
        # 拷贝一个副本
        newWb = copy(oldWb)
        # 从副本中获取当前sheet页
        sheet = newWb.get_sheet(0)
        # 整理采集数据转化成一个list列表
        line = [item['book_name'],item['english_name'],item['author'],item['img'],item['publish'],item['price'],item['score'],item['description']]
        # 循环追加写入excel文件
        for colIndex in range(0, len(item)):
            # 写入电影的bookItem中的每一项数据
            sheet.write(self.rowIndex, colIndex, line[colIndex])
        # 完毕后保存excel,自动覆盖原有文件        	
        newWb.save(self.excelPath)
        # 全局行数自增1
        self.rowIndex = self.rowIndex + 1
        # 返回item
        return item