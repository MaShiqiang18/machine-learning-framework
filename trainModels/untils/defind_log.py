# coding=gbk

import logging
import time


class DefindLog(object):
    # ��װlogging

    def __init__(self, log_name, logger=None):
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.NOTSET)
        self.log_time = time.strftime("%Y_%m_%d_")
        self.log_name = log_name

    def set_logger(self):
        # �ж����handlers����handler������µ�handler
        if not self.logger.handlers:
            # ��������־д�뵽�ļ���a��ʾ��׷�ӵ���ʽд����־
            self.fh = logging.FileHandler(self.log_name, "a")
            self.fh.setLevel(logging.INFO)
            # �����ӿ���̨�����־
            self.chd = logging.StreamHandler()
            # ����Ϊnotset�����Դ�ӡdebug��info��warning��error��critical����־����
            self.chd.setLevel(logging.INFO)

            # ������־�ļ��������ʽ
            # self.formatter = logging.Formatter(
            #     "[%(levelname)s]--%(asctime)s-%(filename)s->%(funcName)s line:%(lineno)d: %(message)s")
            self.formatter_fh = logging.Formatter("[%(levelname)s]--%(asctime)s: %(message)s")
            self.fh.setFormatter(self.formatter_fh)
            # ���ÿ���̨�������ʽ
            self.format_chd = logging.Formatter("[%(levelname)s]--%(asctime)s: %(message)s")
            self.chd.setFormatter(self.format_chd)

            # ����ļ���־����־������
            self.logger.addHandler(self.fh)
            # ��ӿ���̨����־������
            self.logger.addHandler(self.chd)

    def get_logger(self):
        DefindLog.set_logger(self)
        # print self.logger.handlers  ��ӡhandlers�б�
        return self.logger

    def remove_log_handler(self):
        # �Ƴ�handlers�е�Ԫ��
        # �Ƴ�����������ظ���ӡ��ͬ����־
        self.logger.removeHandler(self.fh)
        # �Ƴ�����������ظ���ӡ��ͬ����־
        self.logger.removeHandler(self.chd)
        # �ر���־������
        self.fh.close()
        # �ر���־������
        self.chd.close()


# if __name__ == "__main__":
#     # ��ʽ�Ƿ��ܴ�ӡ�ɹ�
#     test = DefindLog()
#     log = test.get_logger()
#     log.warning("this is warning information")
#     log.info("this is info informattion")
#     log.info("this is info informattion1")
#     log.debug("this is debug information")