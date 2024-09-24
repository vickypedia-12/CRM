import scrapy
import json
from w3lib.html import remove_tags

class ContentExtractorSpider(scrapy.Spider):
    name = "content_extractor"
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'ROBOTSTXT_OBEY': False,
        'RETRY_TIMES': 2,
        'FEED_EXPORT_FIELDS': None,
        'LOG_LEVEL': 'DEBUG',
        'FEED_FORMAT': 'jsonl',
        'FEED_URI': 'content.jsonl',
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',
    }

    def start_requests(self):
         with open('urls.json', 'r') as f:
            urls = json.load(f)
        
            for item in urls:
                if isinstance(item, dict):
                    url = item.get('url')
                else:
                    url = item 
                if url:
                    yield scrapy.Request(url, callback=self.parse)


    def parse(self, response):
        text_content = []
        for selector in ['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table', 'li', 'th','td']:
            elements = response.css(f'{selector}::text').getall()
            text_content.extend([text.strip() for text in elements if text.strip()])

        
        combined_text = ' '.join(text_content)
        yield {
            'url': response.url,
            'content': combined_text
            }

    def closed(self, reason):
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(f"Total pages processed: {self.crawler.stats.get_value('response_received_count')}")