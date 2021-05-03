from multiprocessing.context import Process

from scrapy.crawler import CrawlerProcess

from wildberries.wildberries.spiders.wildberries import WildberriesSpider


def parser(final_link, file_name):
    #file_name = f'{final_link[0][35:-7]}.jl'

    def crawl():
        crawler = CrawlerProcess(settings={
                  "FEEDS": {file_name: {"format": "jl","encoding":"utf-8"},
    },
    },
    )
        crawler.crawl(WildberriesSpider,start_urls = [final_link])
        crawler.start()

    process = Process(target=crawl)
    process.start()
    process.join()

    return file_name
