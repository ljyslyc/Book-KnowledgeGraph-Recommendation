# -*- coding: utf-8 -*-
import scrapy
from book_spider.items import BookItem, BookCommentItem, DoubanItem

class DoubanSpider(scrapy.Spider):
    name = 'douban'
    allowed_domains = ['douban.com']

    def start_requests(self):
        """
        Top250列表页非常规律，每页25条记录，所以后一页查询参数为当前页查询参数+25
        :return:
        """
        # for i in range(0, 250, 25):
        #     yield scrapy.Request('https://book.douban.com/top250?start={}'.format(i), self.parse)

        # 调试阶段，只用一个URL
        yield scrapy.Request('https://book.douban.com/top250?start={}'.format(0), self.parse)

    def parse(self, response):
        """
        解析书籍列表页
        :param response:
        :return:
        """
        for book in response.xpath(".//div[@class='article']//table"):
            item = DoubanItem()
            item['img'] = book.xpath(".//a/img/@src").extract_first()
            item['book_name'] = book.xpath(".//div[@class='pl2']/a/@title").extract_first().strip()
            item['english_name'] = ' '.join(book.xpath(".//div[@class='pl2']/span/text()").extract())
            item['author'] = book.xpath(".//p[@class='pl']/text()").extract_first().split('/')[0]
            item['publish'] = book.xpath(".//p[@class='pl']/text()").extract_first().split('/')[-3]
            item['price'] = book.xpath(".//p[@class='pl']/text()").extract_first().split('/')[-1]
            item['score'] = book.xpath(".//span[@class='rating_nums']/text()").extract_first()
            item['description'] = book.xpath(".//p[@class='quote']/span/text()").extract_first()
            yield item

        # 下一页请求跳转，实现自动翻页
        nextPage = response.xpath('//span[@class="next"]/a/@href')
        # 判断nextPage是否有效（无效代表当前页面为最后一页）
        if nextPage:
            # 获取nextPage中的下一页链接地址并加入到respones对象的请求地址中
            url = response.urljoin(nextPage[0].extract())
            # 发送下一页请求并调用parse()函数继续解析
            yield scrapy.Request(url, self.parse)
        # pass

