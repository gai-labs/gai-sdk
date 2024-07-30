import os
import cmd
import readline


class MultilineInputCmd:
    multiline_string = ""

    # Overwrite prompt attribute
    # prompt = 'locallm> '
    @property
    def prompt(self):
        if self.in_multiline_mode:
            return f'''> '''
        return (
            f'''hot-keys: [^S] summarize, [^P] push, [^R] pop, [^T] speak, [^I] improve, [^L] critique [^E] exit\n'''
            f'''================================================================================================\n'''
            f'''{self.generator}> ''')

    def __init__(self):
        cmd.Cmd.__init__(self)  # initialize the base class
        self.buffer = ""
        readline.parse_and_bind('tab: complete')
        readline.parse_and_bind('\C-s: "summarize\n"')
        readline.parse_and_bind('\C-p: "push\n"')
        readline.parse_and_bind('\C-r: "pop\n"')
        readline.parse_and_bind('\C-h: "history\n"')
        readline.parse_and_bind('\C-t: "speak\n"')
        readline.parse_and_bind('\C-i: "improve\n"')
        readline.parse_and_bind('\C-l: "critique\n"')
        readline.parse_and_bind('\C-e: "exit\n"')
        # readline.parse_and_bind('"\\e[A": prev_command')
        self.multiline_string = ""
        self.in_multiline_mode = False
        self.command = ""

    # def do_prev_command(self, arg):
    #     if readline.get_current_history_length() > 0:
    #         self.onecmd(readline.get_history_item(
    #             readline.get_current_history_length()))

    def cmdloop(self, intro=None):
        self.preloop()
        if intro is not None:
            self.intro = intro
        if self.intro:
            self.stdout.write(str(self.intro) + "\n")
        stop = None
        while not stop:
            if self.cmdqueue:
                line = self.cmdqueue.pop(0)
            else:
                try:
                    if self.in_multiline_mode:
                        line = input('... ')
                        if line.endswith("'") and not line.endswith("\\'"):
                            self.in_multiline_mode = False
                            self.multiline_string += ' ' + line[:-1]
                            line = self.command + ' ' + \
                                self.multiline_string.replace("\\'", "'")
                        else:
                            self.multiline_string += ' ' + line
                            continue
                    else:
                        line = input(self.prompt)
                        if line.count("'") % 2 != 0 and not line.endswith("\\'"):
                            self.in_multiline_mode = True
                            self.command, self.multiline_string = line.rsplit(
                                "'", 1)
                            continue
                except EOFError:
                    line = 'EOF'
            line = self.precmd(line)
            stop = self.onecmd(line)
            stop = self.postcmd(stop, line)
        self.postloop()
