import scrapy

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://tcetmumbai.in/*']

    def parse(self,response):
        for text in response.css('font'):
            yield {'text': text.css('::text').get()}

        for list in response.css('li'):
            yield {'list': list.css('::text').get()}

        
        