import datetime
import os


class Logger:
    PATH = os.getcwd()
    LOG_FILE = None

    @staticmethod
    def init():
        Logger.LOG_FILE = open("log.txt", mode="w", encoding="UTF-8")

    @staticmethod
    def _get_log_file():
        return Logger.LOG_FILE

    @staticmethod
    def print(msg):
        file = Logger._get_log_file()
        if file is not None:
            file.write(str(datetime.datetime.now()) + " - " + msg)
            file.write("\n")

    @staticmethod
    def destroy():
        Logger.LOG_FILE.close()
