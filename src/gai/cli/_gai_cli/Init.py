import os,json
from gai.lib.common.generators_utils import chat_string_to_list
storage = "/tmp/messages.txt"
cmd_storage = "/tmp/cmd_history.txt"


# This dir
import os
this_dir=os.path.dirname(os.path.realpath(__file__))
import sys,json
sys.path.insert(0, this_dir)

# Model
generator = "gpt-4"
name = "Assistant"

# Load world prompt
with open(os.path.join(this_dir, "../sys_prompts", "world_prompt.txt"), "r") as f:
    world_prompt = f.read()
sys_env_message = chat_string_to_list(world_prompt)[0]

# Load role prompt
with open(os.path.join(this_dir, "../sys_prompts", "role_prompt.txt"), "r") as f:
    role_prompt = f.read()
sys_id_message = chat_string_to_list(role_prompt)[0]
sys_id_message['content'] = sys_id_message['content'].format(name=name)

# Load tools
with open(os.path.join(this_dir, "../tools.txt"), "r") as f:
    tools = json.loads(f.read())

class Init:

    def load_storage(self):
        if not os.path.exists(storage):
            return [
                sys_env_message
            ]
        with open(storage, "r") as f:
            j = f.read()
            if j:
                j = json.loads(j)
            if not j:
                return []
            return j

    def load_cmd_history(self):
        if not os.path.exists(cmd_storage):
            return []
        with open(cmd_storage, "r") as f:
            j = f.read()
            if j:
                j = json.loads(j)
            if not j:
                return []
            return j


    def __init__(self):
        self.messages = self.load_storage()
        self.cmd_history = self.load_cmd_history()

    def save(self):
        with open(storage, "w") as f:
            f.write(json.dumps(self.messages))
        with open(cmd_storage, "w") as f:
            f.write(json.dumps(self.cmd_history))

    def do_clear(self,args):
        self.messages = []
        self.save()

    def _get_version(self):
        VERSION = ''
        version_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "../VERSION")
        with open(version_path, "r") as f:
            VERSION = f.read().strip()
        return VERSION

    def do_version(self,ignored):
        VERSION = self._get_version()
        print(VERSION)