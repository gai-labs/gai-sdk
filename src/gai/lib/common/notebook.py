from IPython.display import display, HTML
def print_colored(text, color):
    display(HTML(f'<font color="{color}">{text}</font>'))
def highlight(text):
    display(HTML(f'<font color="yellow">{text}</font>'))
def error(text):
    display(HTML(f'<font color="red">{text}</font>'))