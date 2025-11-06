
import os
import datetime as dt
from typing import Optional

class Logger:
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, "train.log"), "a", encoding="utf-8")
        self.tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=log_dir)
            except Exception:
                self.tb = None

    def info(self, msg: str):
        line = f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_file.write(line + "
")
        self.log_file.flush()

    def add_scalar(self, tag: str, value: float, step: int):
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)

    def close(self):
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        self.log_file.close()
