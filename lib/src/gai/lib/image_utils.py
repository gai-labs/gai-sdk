import base64
import imghdr
from PIL import Image, ImageFilter
from io import BytesIO
from gai.lib.logging import getLogger
logger = getLogger(__name__)

def read_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded_string = base64.b64encode(data)
        base64_image =  encoded_string.decode('utf-8')
        return base64_image

def read_to_base64_imageurl(image_path):
    with open(image_path, "rb") as image_file:
        data = image_file.read()
        encoded_string = base64.b64encode(data)
        base64_image =  encoded_string.decode('utf-8')
        type = imghdr.what(None,data)
        image_url = {
            "url":f"data:image/{type};base64,{base64_image}"
        }
        return image_url

def base64_to_imageurl(base64_image):
    img_data = base64.b64decode(base64_image)
    img_type = imghdr.what(None, img_data)
    image_url = {
        "url":f"data:image/{img_type};base64,{base64_image}"
    }
    return image_url

def bytes_to_imageurl(image_bytes):
    img_type = imghdr.what(None, image_bytes)
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    image_url = {
        "url":f"data:image/{img_type};base64,{base64_image}"
    }
    return image_url

def resize_image(image_bytes:bytes, width:int, height:int, useGaussian:bool=True, imageType:str='PNG', blur_radius:float=0.5):
    image = Image.open(BytesIO(image_bytes))

    if useGaussian:
        # Apply a slight Gaussian blur
        if image.mode != 'RGBA':
                image = image.convert('RGBA')    
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        # Resize the image using a high-quality downsampling filter

    image = image.resize((width, height), Image.LANCZOS)

    thumb_buffer = BytesIO()
    image.save(thumb_buffer, format=imageType,compression_level=9,optimize=True)
    logger.info(f"Resized image from 512x512 to {width}x{height}, original={len(image_bytes)} bytesize={thumb_buffer.tell()}")

    thumb_buffer.seek(0)
    return thumb_buffer.getvalue()

def save_image(image_bytes, file_path):
    image = Image.open(BytesIO(image_bytes))
    image.save(file_path)