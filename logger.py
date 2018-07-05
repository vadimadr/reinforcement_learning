import os
from collections import defaultdict

logger = None


class Logger:
    def __init__(self, backends):
        self.backends = backends


def init_logging(output_dir, backends='log,tb'):
    global logger
    if isinstance(backends, str):
        backends = [make_logging_backend(bck, output_dir) for bck in backends.split(',')]
    logger = Logger(backends)


def make_logging_backend(backend, output_dir):
    if backend.lower() == 'tb':
        return TensorBoardBackend(output_dir)
    if backend.lower() == 'log':
        return LogBackend(output_dir)


class LoggingBackend(object):
    def log_scalar(self, k, v):
        pass

    def log_text(self, text):
        pass


class TensorBoardBackend(LoggingBackend):
    def __init__(self, output_dir):
        os.makedirs(str(output_dir), exist_ok=True)

        from tensorboardX import SummaryWriter
        self.tb = SummaryWriter(output_dir)
        self.scalar_steps = defaultdict(int)

    def log_scalar(self, k, v):
        self.scalar_steps[k] += 1
        self.tb.add_scalar(k, v, self.scalar_steps[k])


class LogBackend(LoggingBackend):
    def __init__(self, output_dir):
        os.makedirs(str(output_dir), exist_ok=True)

    def log_text(self, text):
        print(text)


def log(text):
    assert logger is not None, "Logger is not initialized"
    logger.log_text(text)


def log_scalar(k, v):
    assert logger is not None, "Logger is not initialized"
    logger.log_scalar(k, v)
