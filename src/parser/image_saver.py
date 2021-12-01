############################################################
# Functions are modified from
# https://github.com/dpapathanasiou/pdfminer-layout-scanner
############################################################
import os
from binascii import b2a_hex


###
# Extracting Images
###


def write_file(folder, filename, filedata, flags='w'):
    """Write the file data to the folder and filename combination
    (flags: 'w' for write text, 'wb' for write binary, use 'a' instead of 'w' for append)"""
    result = False
    if os.path.isdir(folder):
        try:
            file_obj = open(os.path.join(folder, filename), flags)
            file_obj.write(filedata)
            file_obj.close()
            result = True
        except IOError:
            pass
    return result


def determine_image_type(stream_first_4_bytes):
    """Find out the image file type based on the magic number comparison of the first 4 (or 2) bytes"""
    file_type = None
    bytes_as_hex = b2a_hex(stream_first_4_bytes)
    if bytes_as_hex.startswith(b'ffd8'):  # modified by Ang: 'ffd8'
        file_type = '.jpeg'
    elif bytes_as_hex == b'89504e47':
        file_type = '.png'
    elif bytes_as_hex == b'47494638':
        file_type = '.gif'
    elif bytes_as_hex.startswith(b'424d'):
        file_type = '.bmp'
    return file_type


def save_image(lt_image, page_number, images_folder):
    """Try to save the image data from this LTImage object, and return the file name, if successful"""
    result = None
    if lt_image['stream']:
        file_stream = lt_image['stream'].get_data()
        if file_stream:
            file_ext = determine_image_type(file_stream[0:4])
            if file_ext:
                file_name = '%d_%s%s' % (
                    page_number, lt_image['name'], file_ext)
                if write_file(images_folder, file_name, file_stream, flags='wb'):
                    result = file_name
    return result
