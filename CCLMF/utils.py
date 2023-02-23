
class Logger():
    def __init__(self):
        self.log_file = None
        self.log_dir = None

    def log(self, str):
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                f.write(str)

    def set_logdir(self, log_dir):
        self.log_dir = log_dir

    def set_filename(self, filename):
        if self.log_dir is not None:
            self.log_file = self.log_dir + filename

    def close(self):
        if self.log_file is not None:
            self.log_file.close()
