import logging
from Algorithm import *

logging.basicConfig(filename='logger.txt', level=logging.INFO)


class Logger(object):
    log_flag = True
    print_flag = True

    def __init__(self):
        pass

    @staticmethod
    def log_now(record):
        if Logger.log_flag:
            logging.info("this iteration loss is:" + str(record.best_loss))
            logging.info("this iteration epoch is:" + str(record.best_epoch))
            logging.info("this iteration costTime is:" + str(record.best_time))
        if Logger.print_flag:
            print("this iteration loss is:" + str(record.best_loss))
            print("this iteration epoch is:" + str(record.best_epoch))
            print("this iteration costTime is:" + str(record.best_time))

    @staticmethod
    def log_best(record):
        if Logger.log_flag:
            logging.info("now best loss is:" + str(record.best_loss))
            logging.info("now best epoch is:" + str(record.best_epoch))
            logging.info("now best costTime is:" + str(record.best_time))
        if Logger.print_flag:
            print("now best loss is:" + str(record.best_loss))
            print("now best epoch is:" + str(record.best_epoch))
            print("now best costTime is:" + str(record.best_time))

    @staticmethod
    def log(msg):
        if Logger.log_flag:
            logging.info(msg)
        if Logger.print_flag:
            print(msg)