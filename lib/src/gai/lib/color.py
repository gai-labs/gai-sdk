class Color:

    COLORS = {
        'white': '\033[97m',
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'purple': '\033[95m',
        'cyan': '\033[96m'
    }
    
    STYLES = {
        'italic': '\033[3m',
        'bold': '\033[1m'
    }
    
    RESET = '\033[0m'

    def __init__(self):
        self._format = []

    def __getattr__(self, attribute):
        if attribute in self.COLORS:
            self._format.append(self.COLORS[attribute])
            return self
        elif attribute in self.STYLES:
            self._format.append(self.STYLES[attribute])
            return self
        else:
            raise AttributeError(f"'Color' object has no attribute '{attribute}'")
    
    def __call__(self, text, end="\n", flush=False):
        formatted_text = "".join(self._format) + text + self.RESET
        print(formatted_text, end=end, flush=flush)
        self._format = []  # Reset for next use        

def yellow(text):
    print(f"\033[33m{text}\033[0m")
def green(text):
    print(f"\033[32m{text}\033[0m")
def red(text):
    print(f"\033[31m{text}\033[0m")