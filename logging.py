import sys, os


class StdoutRedirector:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
        self.is_progress_bar = False

    def write(self, message):
        self.terminal.write(message)
        if '━' in message:
            self.is_progress_bar = True
            return
        if self.is_progress_bar and ('/' in message or message.strip().isdigit()):
            return
        if self.is_progress_bar and message.strip() == '':
            self.is_progress_bar = False
        if not self.is_progress_bar:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging():
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, "_results.txt")