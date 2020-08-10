# coding=gbk

import logging
import time


class DefindLog(object):
    # 封装logging

    def __init__(self, log_name, logger=None):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.NOTSET)
        self.log_time = time.strftime("%Y_%m_%d_")
        self.log_name = log_name

    def set_logger(self):
        # 判断如果handlers中无handler则添加新的handler
        if not self.logger.handlers:
            # 创建将日志写入到文件，a表示以追加的形式写入日志
            self.fh = logging.FileHandler(self.log_name, "a")
            self.fh.setLevel(logging.INFO)
            # 创建从控制台输出日志
            self.chd = logging.StreamHandler()
            # 设置为notset，可以打印debug、info、warning、error、critical的日志级别
            self.chd.setLevel(logging.INFO)

            # 设置日志文件中输出格式
            # self.formatter = logging.Formatter(
            #     "[%(levelname)s]--%(asctime)s-%(filename)s->%(funcName)s line:%(lineno)d: %(message)s")
            self.formatter_fh = logging.Formatter("[%(levelname)s]--%(asctime)s: %(message)s")
            self.fh.setFormatter(self.formatter_fh)
            # 设置控制台中输出格式
            self.format_chd = logging.Formatter("[%(levelname)s]--%(asctime)s: %(message)s")
            self.chd.setFormatter(self.format_chd)

            # 添加文件日志的日志处理器
            self.logger.addHandler(self.fh)
            # 添加控制台的日志处理器
            self.logger.addHandler(self.chd)

    def get_logger(self):
        DefindLog.set_logger(self)
        # print self.logger.handlers  打印handlers列表
        return self.logger

    def remove_log_handler(self):
        # 移除handlers中的元素
        # 移除句柄，避免重复打印相同的日志
        self.logger.removeHandler(self.fh)
        # 移除句柄，避免重复打印相同的日志
        self.logger.removeHandler(self.chd)
        # 关闭日志处理器
        self.fh.close()
        # 关闭日志处理器
        self.chd.close()


# if __name__ == "__main__":
#     # 调式是否能打印成功
#     test = DefindLog()
#     log = test.get_logger()
#     log.warning("this is warning information")
#     log.info("this is info informattion")
#     log.info("this is info informattion1")
#     log.debug("this is debug information")