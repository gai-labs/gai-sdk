import base64
import imghdr
from PIL import Image, ImageFilter
from io import BytesIO
from gai.lib.common.logging import getLogger
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

def bytes_to_imageurl(img_data):
    img_type = imghdr.what(None, img_data)
    base64_image = base64.b64encode(img_data).decode('utf-8')
    image_url = {
        "url":f"data:image/{img_type};base64,{base64_image}"
    }
    return image_url

def resize_image(image_bin, width, height, useGaussian=True, imageType='PNG', blur_radius=0.5):
    image = Image.open(BytesIO(image_bin))

    if useGaussian:
        # Apply a slight Gaussian blur
        if image.mode != 'RGBA':
                image = image.convert('RGBA')    
        image = image.filter(ImageFilter.GaussianBlur(blur_radius))
        # Resize the image using a high-quality downsampling filter

    image = image.resize((width, height), Image.LANCZOS)

    thumb_buffer = BytesIO()
    image.save(thumb_buffer, format=imageType,compression_level=9,optimize=True)
    logger.info(f"Resized image from 512x512 to {width}x{height}, original={len(image_bin)} bytesize={thumb_buffer.tell()}")

    thumb_buffer.seek(0)
    return thumb_buffer.getvalue()

def save_image(image_bin, file_path):
    image = Image.open(BytesIO(image_bin))
    image.save(file_path)