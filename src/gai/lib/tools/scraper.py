from gai.lib.common import http_utils
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from requests.exceptions import RequestException
from urllib.parse import urljoin, urlparse

class Scraper:

    # Scrape and clean the text from the URL
    def scrape(url):

        def _get_base_url(url):
            parsed_url = urlparse(url)
            base_url = "{uri.scheme}://{uri.netloc}".format(uri=parsed_url)
            return base_url


        if not http_utils.is_url(url):
            raise ValueError("Please provide a valid URL")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}             # Some websites block requests that don't have a user agent
            response = requests.get(url,headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses

        except RequestException as e:  # This will handle any requests exceptions including HTTPError
            status_code = e.response.status_code if e.response else 500
            error_msg = f"Scraper.scrape: Remote site error. status_code={str(status_code)} error={type(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        except Exception as e:  # Catch any other exception that can be thrown by requests
            error_msg = f"Scraper.scrape: Non Remote site error. error={type(e)}"
            logger.error(error_msg)

        soup = None
        try:
            html_text = response.text
            soup = BeautifulSoup(html_text, "html.parser")
        except Exception as e:
            error_msg = f"Scraper.scrape: HTML parsing error. error={type(e)}"
            raise Exception(error_msg)

        # remove all javascript and stylesheet code and get the text
        for script in soup(["script", "style"]): # remove all javascript and stylesheet code
            script.extract()
        cleaned_text = ' '.join(soup.stripped_strings)

        # extract all the links
        base_url = _get_base_url(url)
        links = soup.find_all('a')
        link_dict = {}
        for link in links:
            href = link.get('href')
            full_url = urljoin(base_url, href)
            text = link.get('title')
            if not text:
                text = link.text.strip()
            if text:
                link_dict[text] = full_url

        return cleaned_text, link_dict
