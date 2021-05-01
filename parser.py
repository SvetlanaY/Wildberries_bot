from scrapy.crawler import CrawlerProcess
from wildberries.wildberries.spiders.wildberries import WildberriesSpider

def parser(final_linc):
    process = CrawlerProcess(settings={
    "FEEDS": {
        "items.jl": {"format": "jl","encoding":"utf-8"},
    },
},start_urls=final_linc)

    process.crawl(WildberriesSpider)
    process.start()
    process.stop()