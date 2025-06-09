from gai.lib.errors import ApiException
from urllib.parse import urlparse
import os
import pprint
import re
import httpx
import requests
import json
from gai.lib.logging import getLogger
logger = getLogger(__name__)


def is_url(s):
    return re.match(r'^https?:\/\/.*[\r\n]*', s) is not None

# Check if URL contains a file extension (e.g. .pdf, .jpg, .png, etc.)

def has_extension(url):
    parsed_url = urlparse(url)
    _, ext = os.path.splitext(parsed_url.path)
    return bool(ext)

import json

def _handle_error_data(error_data, status_code):
    error_code = "unknown"
    if isinstance(error_data, str):
        raise ApiException(status_code=status_code, code=error_code, message=error_data)

    if 'detail' in error_data:
        if isinstance(error_data['detail'], str):
            raise ApiException(status_code=status_code, code=error_code, message=error_data['detail'])

        if 'code' in error_data['detail']:
            error_code = error_data['detail']['code']

        if 'message' in error_data['detail'] and isinstance(error_data['detail']['message'], str):
            raise ApiException(status_code=status_code, code=error_code, message=error_data['detail']['message'])

    if 'code' in error_data:
        error_code = error_data['code']
        if 'message' in error_data:
            raise ApiException(status_code=status_code, code=error_code, message=error_data['message'])

    raise ApiException(status_code=status_code, code=error_code, message=json.dumps(error_data))

def _handle_failed_response_sync(response : requests.Response):
    if response.status_code == 401:
        raise ApiException(status_code=401, code="unauthorized", message="Unauthorized")
    content_type = response.headers.get("Content-Type")
    if content_type and "application/json" in content_type:
        error_data = response.json()
    else:
        if isinstance(response.text, str):
            error_data = response.text
        else:
            error_data = response.text()
    _handle_error_data(error_data, response.status_code)

def http_post(url, data=None, files=None,timeout=30.0):
    if data == None and files == None:
        raise Exception("No data or files provided")
    logger.debug(f"httppost:url={url}")
    logger.debug(f"httppost:data={pprint.pformat(data)}")
    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }

    try:

        if files:
            if data and "stream" in data:
                files["stream"] = (None, data["stream"])
            response = requests.post(url, files=files,verify=verify_ssl)
        else:
            if "stream" in data:
                try:
                    response = requests.post(url, json=data, stream=data["stream"], headers=headers,verify=verify_ssl,timeout=timeout)
                except Exception as e:
                    logger.error(f"Exception in http_post: {url} {str(e)}")
                    raise
            else:
                try:
                    response = requests.post(url, json=data,verify=verify_ssl,timeout=timeout)
                except Exception as e:
                    logger.error(f"Exception in http_post: {url} {str(e)}")
                    raise
        if response.status_code == 200:
            return response
        else:
            _handle_failed_response_sync(response)

    except requests.exceptions.ConnectionError as e:
        raise Exception("Connection Error. Is the service Running?")

def http_get(url):
    logger.debug(f"httppost:url={url}")
    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }

    try:

        response = requests.get(url, headers=headers,verify=verify_ssl)
        if response.status_code == 200:
            return response
        else:
            _handle_failed_response_sync(response)
    except requests.exceptions.ConnectionError as e:
        raise Exception("Connection Error. Is the service Running?")

def http_delete(url):
    logger.debug(f"httppost:url={url}")
    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }

    try:
        response = requests.delete(url, headers=headers,verify=verify_ssl)
        if response.status_code == 200:
            return response
        else:
            _handle_failed_response_sync(response)
    except requests.exceptions.ConnectionError as e:
        raise Exception("Connection Error. Is the service Running?")

def http_put(url):
    logger.debug(f"httppost:url={url}")
    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }
    
    try:
        response = requests.put(url, headers=headers, verify=verify_ssl)
        if response.status_code == 200:
            return response
        else:
            _handle_failed_response_sync(response)
    except requests.exceptions.ConnectionError as e:
        raise Exception("Connection Error. Is the service Running?")

async def _handle_failed_response_async(response : httpx.Response):
    if response.status_code == 401:
        raise ApiException(status_code=401, code="unauthorized", message="Unauthorized")
    content_type = response.headers.get("Content-Type")
    if content_type and "application/json" in content_type:
        error_data = response.json()
    else:
        error_data = response.text
    _handle_error_data(error_data, response.status_code)

async def http_post_async(url, data=None, files=None, timeout:float=120.0):
    if data == None and files == None:
        raise Exception("No data or files provided")

    logger.debug(f"httppost:url={url}")
    logger.debug(f"httppost:data={pprint.pformat(data)}")
    logger.debug(f"httppost:files={pprint.pformat(files)}")

    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }

    async with httpx.AsyncClient(timeout=timeout,verify=verify_ssl,) as client:
        try:
            if files:
                if data and "stream" in data:
                    files["stream"] = (None, data["stream"])
                response = await client.post(url, files=files, headers=headers )
            else:
                if "stream" in data:
                    response = await client.post(url, json=data, stream=data["stream"], headers=headers)
                else:
                    response = await client.post(url, json=data, headers=headers)

            if response.status_code == 200:
                return response
            else:
                await _handle_failed_response_async(response)

        except httpx.ConnectError as e:
            raise Exception("Connection Error. Is the service Running?")
        except Exception as e:
            logger.error(f"http_utils.http_post_async: {str(e)}")
            raise
        

async def http_get_async(url,data=None, timeout:float=120.0):
    logger.debug(f"httppost:url={url}")
    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }

    async with httpx.AsyncClient(verify=verify_ssl,timeout=timeout) as client:
        try:
            if data:
                headers["Content-Type"] = "application/json"
                response = await client.request(method='GET',url=url, content=json.dumps(data), headers=headers)
            else:
                response = await client.get(url,headers=headers)
            if response.status_code == 200:
                return response                      # Returning the data
            else:
                await _handle_failed_response_async(response)
        except httpx.HTTPStatusError as e:
            raise Exception("Connection Error. Is the service Running?")
        except Exception as e:
            logger.error(f"http_utils.http_get_async: {str(e)}")
            raise

async def http_delete_async(url,timeout:float=120.0):
    logger.debug(f"httppost:url={url}")
    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }

    async with httpx.AsyncClient(verify=verify_ssl,timeout=timeout) as session:
        try:
            response = await session.delete(url,headers=headers)
            if response.status_code == 200:
                return response
            else:
                await _handle_failed_response_async(response)
        except httpx.HTTPStatusError as e:
            raise Exception("Connection Error. Is the service Running?")
        except Exception as e:
            logger.error(f"http_utils.http_delete_async: {str(e)}")
            raise
    
async def http_put_async(url, data=None,timeout:float=120.0):
    logger.debug(f"httppost:url={url}")
    # Disable SSL verification if URL is for localhost
    verify_ssl = not (url.startswith('https://localhost') or url.startswith('https://127.0.0.1'))
    gai_api_key=os.environ.get("GAI_API_KEY", None)
    headers = {}
    if gai_api_key:
        headers = {
            "X-Api-Key": gai_api_key
        }

    async with httpx.AsyncClient(verify=verify_ssl,timeout=timeout) as session:
        try:
            response = await session.put(url,headers=headers,json=data)
            if response.status_code == 200:
                return response
            else:
                await _handle_failed_response_async(response)
        except httpx.HTTPStatusError as e:
            raise Exception("Connection Error. Is the service Running?")
        except Exception as e:
            logger.error(f"http_utils.http_put_async: {str(e)}")
            raise
