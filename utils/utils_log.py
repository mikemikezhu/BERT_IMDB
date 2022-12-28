import logging

from decorator.singleton import Singleton
from utils.utils_time import TimeUtils
from utils.utils_pid import PidUtils
from utils.constants import *

import os


@Singleton
class LogUtils:

    def __init__(self):

        if not os.path.exists(LOG_PATH):
            raise Exception(
                "Please create \"log\" folder under root path before running the program!")

        self._extra = {'pid': PidUtils.instance().get_pid()}
        formatter = logging.Formatter(
            '%(asctime)s %(pid)s %(levelname)s %(message)s')
        start_time = TimeUtils.instance().get_start_time()

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        default_handler = logging.FileHandler(
            "{}/{}_log_general.log".format(LOG_PATH, start_time))
        default_handler.setFormatter(formatter)

        default_logger = logging.getLogger(LOGGER_DEFAULT)
        default_logger.addHandler(default_handler)
        default_logger.addHandler(stream_handler)

    def log_info(self, msg, type=LOGGER_DEFAULT):
        logger = logging.getLogger(type)
        logger = logging.LoggerAdapter(logger, self._extra)
        logger.setLevel(logging.INFO)
        logger.info(msg)

    def log_warning(self, msg, type=LOGGER_DEFAULT):
        logger = logging.getLogger(type)
        logger = logging.LoggerAdapter(logger, self._extra)
        logger.setLevel(logging.WARN)
        logger.warn(msg)

    def log_error(self, msg, type=LOGGER_DEFAULT):
        logger = logging.getLogger(type)
        logger = logging.LoggerAdapter(logger, self._extra)
        logger.setLevel(logging.ERROR)
        logger.error(msg)
