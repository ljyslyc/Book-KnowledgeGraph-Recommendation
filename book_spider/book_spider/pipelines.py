# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


class BookSpiderPipeline(object):
    def process_item(self, item, spider):
        print('book_name:' + item['book_name'])
        print('author:' + item['author'])
        print('img:' +item['img'])
        print('english_name:' + item['english_name'])
        print('publish:' + item['publish'])
        print('price:' + item['price'])
        print('score:' + item['score'])
        print('description:' + item['description'])
        print('>>>')
        return item
