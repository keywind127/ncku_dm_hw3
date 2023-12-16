from datetime import datetime

from typing import *

class StopWatch(object):

    def __init__(self, fmt_str : Optional[ str ] = "Total Time: {}") -> None:
        self.fmt_str = fmt_str

    def __enter__(self, *args, **kwargs) -> None:
        self.start_of_time = datetime.now()

    def __exit__(self, *args, **kwargs) -> None:
        print(self.fmt_str.format((datetime.now() - self.start_of_time).total_seconds()))


if (__name__ == "__main__"):

    with StopWatch("Haha: {}") as a:

        __import__("time").sleep(3.1415)