import datetime

from emotions.detecting.Constants import LOGGER_FILE


class Logger:
    LOG_FILE = None

    @staticmethod
    def init():
        Logger.LOG_FILE = open(LOGGER_FILE, mode="w", encoding="UTF-8")

    @staticmethod
    def print(msg):
        file = Logger.LOG_FILE
        if file is not None:
            file.write("{time} - {message}\n".format(time=datetime.datetime.now(), message=msg))

    @staticmethod
    def destroy():
        Logger.LOG_FILE.close()
