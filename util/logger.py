import os
import sys
import time
import logging


def init_logger(log_file_save_path: str) -> logging.Logger:
    """
    初始化日志对象
    :return: 日志对象
    """

    log_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_file_save_path = os.path.join(log_file_save_path, log_time + '.log')

    if not os.path.exists(os.path.dirname(log_file_save_path)):
        os.makedirs(os.path.dirname(log_file_save_path))

    log_handler = logging.FileHandler(log_file_save_path, encoding='utf-8')
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S'
    )
    logging.root.setLevel(level=logging.INFO)
    logger.addHandler(log_handler)

    logger.info('log_file_save_path: {}'.format(log_file_save_path))

    return logger


def create_log_file(log_file_save_path: str) -> open:
    """
    创建日志文件
    :param log_file_save_path: 日志文件目录
    :return: 日志文件对象
    """

    if not os.path.exists(os.path.dirname(log_file_save_path)):
        os.makedirs(os.path.dirname(log_file_save_path))
    log_file = open(log_file_save_path, 'w', encoding='utf-8')
    return log_file
