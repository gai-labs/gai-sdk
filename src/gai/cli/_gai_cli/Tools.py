from gai.lib.tools.scraper import Scraper
from gai.lib.tools.googler import Googler
from rich.console import Console
console=Console()
gg=Googler()

class Tools:

    def __init__(self,client):
        self.client = client

    def gg(self, search_term,max_results):
        results = gg.google(search_term)
        inner_messages=[]
        inner_messages.append({ "role":"user", "content":f"gg {search_term}"})
        inner_messages.append({ "role":"assistant", "content":results})
        self.save()
        n_results = 0
        summaries = []
        for result in results:
            if n_results > max_results:
                break
            try:
                console.print(f"[yellow]Scraping {result['url']}...[/]")
                content, links = Scraper().scrape(result["url"])
                messages= [
                        {"role": "system", "content": """You are an expert in summarizing <content> provided by the user that is scraped from the web and convert into point form summaries.
                        Follow this steps:
                        1) Ignore non-relevant, advertisements contents as well as content that describes the website instead of relevant to the user's query. 
                        2) Proofread and summarise the content relevant to the user's search.
                        3) Present the summary in point form."""},
                        {"role": "user",
                            "content": f"Summarize this <content>{content}</content>"},
                        {"role": "assistant", "content": ""},
                    ]
                summaries.append( self.client.chat.completions.create(model="exllamav2-mistral7b",messages=messages,stream=False).extract() )
                n_results += 1
            except:
                continue        
        
        messages=[
            {"role": "user", "content": f"Extact, proofread and summarize <content>{str(summaries)}</content> that is relevant to {search_term} into point forms."},
            {"role": "assistant", "content": ""},
        ]
        content=""
        for chunk in self.client.chat.completions.create(model="exllamav2-mistral7b",messages=messages):
            chunk = chunk.extract()
            if (chunk) and type(chunk)==str:            
                content+=chunk
                console.print(f"[italic bright_white]{chunk}[/]",end="")

        print()
        inner_messages.append({ "role":"assistant", "content":content})
        return inner_messages

    def do_gg(self, arg):
        inner_messages = self.gg(
            search_term=arg, 
            max_results=5
            )
        for inner_message in inner_messages:
            self.messages.append(inner_message)
        self.save()

    def scrape(self,url):
        result, links = Scraper.scrape(url=url)
        inner_messages=[]
        inner_messages.append({"role":"user","content":f"scrape {url}"})
        inner_messages.append({"role":"assistant","content":result})
        inner_messages.append({"role":"system","content":"Remove meaningless sentences from the above result and organise them into coherent paragraphs."})
        inner_messages.append({"role":"assistant","content":""})
        content=""
        for chunk in self.client.chat.completions.create(model="exllamav2-mistral7b",messages=inner_messages):
            chunk = chunk.extract()
            if (chunk) and type(chunk)==str:
                content+=chunk
                print(chunk, end="", flush=True)
        print()
        inner_messages.append({"role": "assistant", "content": content})
        return inner_messages

    def do_scrape(self, arg):
        inner_messages=self.scrape(
            url=arg
            )
        for inner_message in inner_messages:
            self.messages.append(inner_message)
        self.save()

    def news(self,url=None):
        if not url:
            url="https://www.bbc.com/news/world"
            #url="https://asiaone.com"
        content=""
        with console.status("Working...",spinner="monkey") as status:
            result, links = Scraper.scrape(url)
            inner_messages=[]
            inner_messages.append({"role":"user","content":f"scrape {url}"})
            inner_messages.append({"role":"assistant","content":result})
            inner_messages.append({"role":"user","content":"Remove meaningless sentences from the above result and organise them into coherent paragraphs. Excluding marketing information about the news portal."})
            inner_messages.append({"role":"assistant","content":""})
            #
            response=self.client.chat.completions.create(
                model="exllamav2-mistral7b",
                messages=inner_messages,
                stream=True,
                max_new_tokens=1000,
                stopping_conditions=[""]
                )
            for chunk in response:
                # Stop the spinner on the first iteration
                if 'status' in locals():
                    status.stop()
                    del status                
                chunk = chunk.extract()
                if chunk and isinstance(chunk, str):
                    
                    # Skip empty chunks
                    if content is None and chunk==" ":
                        continue

                    content += chunk
                    console.print(f"[italic bright_white]{chunk}[/]",end="")
            print()
            inner_messages.append({"role": "assistant", "content": content})
            return inner_messages

    def do_news(self,arg):
        inner_messages=self.news(
            url=arg
            )
        for inner_message in inner_messages:
            self.messages.append(inner_message)
        self.save()
      
    def do_bus(self,arg):
        pass

