import datetime


class Logger:

    @staticmethod
    def print(msg):
        print(str(datetime.datetime.now()) + " - " + msg)
