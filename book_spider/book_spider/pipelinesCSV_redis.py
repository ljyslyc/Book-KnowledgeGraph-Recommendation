import csv,re
import itertools
import time
 
class DoubanbookPipeline(object):

    # __init__()函数为该类的构造函数，主要实现创建一个csv文件
    def __init__(self):
        # 创建输出文件夹名称
        folder_name = 'RedisData'
        # 获取系统当前时间
        now = time.strftime('%Y-%m-%d', time.localtime())

        # 书籍信息
        # 设置保存文件名称
        bookfileName = 'book' + now + '.csv'
        # 最终文件路径
        self.bookPath = folder_name + '/' + bookfileName      
        # 创建csv
        self.csvbookwriter = csv.writer(open(self.bookPath, 'a+', encoding='utf-8',newline=''), delimiter=',')
        self.csvbookwriter.writerow(['book_number','book_name','cover','authors','press','publish_year',
                                'price','score','tags'])

        # 评论信息
        # 设置保存文件名称
        comfileName = 'com' + now + '.csv'
        # 最终文件路径
        self.comPath = folder_name + '/' + comfileName      
        # 创建csv
        self.csvcomwriter = csv.writer(open(self.comPath, 'a+', encoding='utf-8',newline=''), delimiter=',')
        self.csvcomwriter.writerow(['book_number','nickname','star','votes','content'])

    # def close_spider(self,spider):
    #     #关闭爬虫时顺便将文件保存退出
    #     self.csvbookwriter.close()
    #     self.csvcomwriter.close()

    def process_item(self, item, ampa):

        if 'book_name' in item:
            book_number  = item['book_number'] if 'book_number' in item else ''
            book_name  = item['book_name'] if 'book_name' in item else ''
            cover  = item['cover'] if 'cover' in item else ''
            press  = item['press'] if 'press' in item else ''
            publish_year  = item['publish_year'] if 'publish_year' in item else ''
            price  = item['price'] if 'price' in item else ''
            score  = item['score'] if 'score' in item else ''
            authors = '|'.join(item['authors']) if 'authors' in item else ''
            tags = '|'.join(item['tags']) if 'tags' in item else ''

            rows1 = [book_number,book_name,cover,authors,press,publish_year,
                        price,score,tags]
            self.csvbookwriter.writerow(rows1)

        if 'nickname' in item:
            book_number  = item['book_number'] if 'book_number' in item else ''
            nickname  = item['nickname'] if 'nickname' in item else ''
            star  = item['star'] if 'star' in item else ''
            votes  = item['votes'] if 'votes' in item else ''
            # 只保留汉子
            content  = re.sub("([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", item['content']) if 'content' in item else ''

            rows1 = [book_number,nickname,star,votes,content]
            self.csvcomwriter.writerow(rows1)
 
        return item