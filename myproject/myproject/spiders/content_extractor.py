import scrapy
import json
from w3lib.html import remove_tags

class ContentExtractorSpider(scrapy.Spider):
    name = "content_extractor"
    extracted_content = []
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': False,
        'RETRY_TIMES': 2,
        'FEED_EXPORT_FIELDS': None,
        'LOG_LEVEL': 'DEBUG',
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',
    }

    def __init__(self, urls, *args, **kwargs):
        super(ContentExtractorSpider, self).__init__(*args, **kwargs)
        self.start_urls = urls
        self.extracted_content = []

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        text_content = []
        for selector in ['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'li', 'th','td']:
            elements = response.css(f'{selector}::text').getall()
            text_content.extend([text.strip() for text in elements if text.strip()])

        combined_text = ' '.join(text_content)
        self.extracted_content.append({
            'url': response.url,
            'content': combined_text
        })
