from logging import debug
from typing import no_type_check

from pdfplumber import page
from .table_extractor import get_raw_tables
from .coord_trans import miner2img, isvisible
from pdfminer.layout import (LTTextBoxHorizontal,
                             LTTextLineHorizontal,
                             LTRect,
                             LTCurve,
                             LTAnno)
from operator import attrgetter
from .image_saver import save_image

import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from PIL import ImageDraw
import os

# TODO: check buttons
# TODO: check buttons in text
# TODO: check layout


class PageParser(object):
    def __init__(self, page, pid_h=150, margins=[15, 15, 0, 0]):
        self.margins = margins  # hyper-parameter
        self.pid_h = pid_h

        self.data = []
        self.page = page
        self.pid = page.page_number
        self.px0, self.py0, self.px1, self.py1 = \
            page.cropbox[0], page.cropbox[1],\
            page.cropbox[2], page.cropbox[3]
        self.pw = float(self.px1 - self.px0)
        self.ph = float(self.py1 - self.py0)
        self.pbox = page.cropbox
        self.middle = (self.pw + 2) / 2

        self.raw_tables = get_raw_tables(page)
        self.blocks = page._layout._objs
        self.text_lines, self.drawings =\
            self.get_elements()  # break blocks into lines

        self.buttons = []
        self.buttons_detected = False

    # ============================================
    # functions for text extraction and formatting
    # ============================================
    def get_text(self):

        #  1. Extract buttons
        self.buttons_detected = self.detect_buttons()

        # 2. Merge items at the same line/row
        merged_lines = self.merge_lines()

        # 3. merge lines into blocks
        merged_blocks = self.merge_blocks(merged_lines)

        return merged_blocks

    def merge_page(self, text_lines, tables):
        page_sequence = ''
        for key in ['left', 'right']:
            itable = 0
            for block in text_lines[key].values():
                page_sequence, itable = \
                    self.insert_table(
                        block[0].y0, tables[key], itable, page_sequence)
                for line in block:
                    page_sequence += line.get_text().replace('\n', '')

            # After merge all the text lines
            line_y0 = self.py1
            page_sequence, itable = \
                self.insert_table(line_y0, tables[key], itable, page_sequence)

        return page_sequence

    def get_sequences(self, text_lines, tables):
        """
        Get sequences in the reading order. For each page,
        sequences are not combined to a complete sequence.
        """
        sequences = []
        for key in ['left', 'right']:
            itable = 0
            for block in text_lines[key].values():
                # check if there is any table before text lines
                table_seqs, itable = \
                    self.check_table_seq(
                        block[0].y0, tables[key], itable)
                sequences += table_seqs

                # combine lines in the block
                seq = ''
                for line in block:
                    seq += line.get_text().replace('\n', '')
                sequences.append(seq)

            # After merge all the text lines, check if there is still table
            line_y0 = self.py1
            table_seqs, itable = \
                self.check_table_seq(
                    line_y0, tables[key], itable)
            sequences += table_seqs

        return sequences

    def check_table_seq(self, line_y0, tables, itable):
        table_seqs = []
        for table_y0, table in tables[itable:]:
            if table_y0 < line_y0:
                len_t = len(table['content'])
                len_c = len(table['content'][0])
                start_row = 0 if len_t == 1 or len_c == 1 else 1
                seq = ''
                for row in table['content'][start_row:]:
                    for item in row:
                        seq += item.replace('\n', '')
                itable += 1
                table_seqs.append(seq)
            else:
                break
        return table_seqs, itable

    def get_elements(self):
        """
        Extract lines and drawings, and separate them into left
        and right.
        The origin of coordinates of blocks/lines/chars are converted from left-bottom to left-top.
        """

        text_lines = {'left': [], 'right': []}
        drawings = {'left': [], 'right': []}

        #  double_objs = []
        middle = (self.pw - 1) / 2

        binary_img = np.zeros((int(self.ph), int(self.pw)))

        self.blocks = sorted(self.blocks, key=attrgetter('y0'), reverse=True)
        itext = 0
        for iblock, block in enumerate(self.blocks):
            bbox = block.bbox

            if isinstance(block, LTTextBoxHorizontal):
                if block.y1 < self.pid_h and block.y1 > self.py0:
                    self.pid = int(block.get_text().strip('\n'))
                    continue
                if not isvisible(bbox, self.pbox, margins=self.margins):
                    continue

                bbox = miner2img(bbox, self.pbox)
                if self.intables(bbox):
                    continue
                block.set_bbox(bbox)

                for line in block._objs:
                    bbox = line.bbox
                    bbox = miner2img(bbox, self.pbox)
                    line.set_bbox(bbox)
                    for char in line:
                        if isinstance(char, LTAnno):
                            continue
                        cbbox = char.bbox
                        cbbox = miner2img(cbbox, self.pbox)
                        char.set_bbox(cbbox)

                    line_dict = {'block_id': iblock,
                                 'element': line}

                    if (bbox[0] >= middle) and itext > 0:
                        text_lines['right'].append(line_dict)
                    else:
                        text_lines['left'].append(line_dict)

                    itext += 1

            elif (isinstance(block, LTRect) or
                  isinstance(block, LTCurve)) and\
                    isvisible(bbox, self.pbox, margins=self.margins):
                # TODO: set hp
                if block.bbox[2] - block.bbox[0] > 140:
                    continue
                bbox = miner2img(bbox, self.pbox)

                dx0 = round(bbox[0])
                dy0 = round(bbox[1])
                dx1 = round(bbox[2])
                dy1 = round(bbox[3])

                # exclude the regions that contain no
                # optical features (all the same value)
                binary_img[dy0:dy1, dx0:dx1] = 1
                '''block.set_bbox(bbox)
                if bbox[0] >= middle:
                    drawings['right'].append(block)
                else:
                    drawings['left'].append(block)'''

        regions = label(binary_img)
        # plt.imshow(regions)
        # plt.show()
        regions = regionprops(regions)
        for region in regions:
            bbox = [region.bbox[1], region.bbox[0],
                    region.bbox[3], region.bbox[2]]
            block = LTRect(0, bbox)
            if bbox[0] >= middle:
                drawings['right'].append(block)
            else:
                drawings['left'].append(block)

        # sort
        for col in ['left', 'right']:
            text_lines[col] = sorted(
                text_lines[col], key=lambda x: x['element'].x0)
            text_lines[col] = sorted(
                text_lines[col], key=lambda x: x['element'].y0)
            drawings[col] = self.sort_objs(drawings[col])

        return text_lines, drawings

    def find_left_right_columns(self):
        left_blocks = []
        right_blocks = []
        #  double_objs = []
        middle = (self.pw - 1) / 2

        for block in self.blocks:
            bbox = block.bbox
            if isinstance(block, LTTextBoxHorizontal) and\
                    isvisible(bbox, self.pbox, margins=self.margins):
                bbox = miner2img(bbox, self.pbox)
                if self.intables(bbox):
                    continue
                block.set_bbox(bbox)

                if bbox[0] >= middle:
                    # right_objs.append((obj_bbox[0], obj_bbox[1], obj))
                    right_blocks.append(block)
                else:
                    left_blocks.append(block)

        return left_blocks, right_blocks

    def intables(self, bbox):
        x0, y0, x1, y1 = bbox
        for table in self.raw_tables:
            if x0 >= table['x0'] and\
                    y0 >= table['y0'] and\
                    x1 <= table['x1'] and\
                    y1 <= table['y1']:
                return True
        return False

    def sort_objs(self, objs):
        '''sort the objs by line first (from top to bottom)
             then sort by horizontal positions'''
        sorted_objs = sorted(objs, key=attrgetter('x0'))
        sorted_objs = sorted(sorted_objs, key=attrgetter('y0'))
        '''sorted_objs = sorted(objs, key=lambda x: x['element'].x0)
        sorted_objs = sorted(sorted_objs, key=lambda x: x['element'].y0)'''

        return sorted_objs

    def detect_buttons(self):

        for key in ['left', 'right']:
            lines_num = len(self.text_lines[key])
            checked_line = 0
            for (i, drawing) in enumerate(self.drawings[key]):
                # TODO: SET Hyperparameter
                is_button = False
                search_start = max(0, checked_line-5)
                for iline in range(search_start, lines_num, 1):
                    text_line = self.text_lines[key][iline]['element']
                    # TODO: SET Hyperparameter
                    # if (drawing.hoverlap(text_line) >= 5) and\
                    #        (drawing.voverlap(text_line) >= 5):
                    text_line_area = text_line.height*text_line.width
                    drawing_area = drawing.height * drawing.width
                    if min(text_line_area, drawing_area) /\
                            max(text_line_area, drawing_area) > 0.4:
                        found_char = False
                        for ichar in range(len(text_line)):
                            char = text_line._objs[ichar]
                            if isinstance(char, LTAnno):
                                continue
                            if (drawing.hoverlap(char) >= 5) and\
                               (drawing.voverlap(char) >= 5):
                                char._text = '[' + char._text
                                found_char = True
                                is_button = True
                                break
                        for jchar in range(len(text_line)-1, ichar-1, -1):
                            char = text_line._objs[jchar]
                            if isinstance(char, LTAnno):
                                continue
                            if (drawing.hoverlap(char) >= 5) and\
                               (drawing.voverlap(char) >= 5):
                                char._text += ']'
                                found_char = True
                                is_button = True
                                break

                        if found_char:
                            checked_line = iline
                            break

                self.buttons.append(is_button)

        return True

    def merge_lines(self):
        """ For each of the line, check if it is in
        the same line with the others. If yes, merge."""
        merged_lines = {}
        for key in ['left', 'right']:
            merged_lines[key] = []
            lines_num = len(self.text_lines[key])
            if lines_num == 0:
                continue
            elif lines_num == 1:
                merged_lines[key] = self.text_lines[key]
                continue
            i = 0
            while(1):
                if i >= lines_num:  # no need for the last
                    break

                curt_iblock = self.text_lines[key][i]['block_id']
                curt_line = self.text_lines[key][i]['element']

                if i + 1 >= lines_num:  # no need for the last
                    merged_line_dict = {'block_id': curt_iblock,
                                        'element': curt_line}

                    merged_lines[key].append(merged_line_dict)
                    break
                next_line = self.text_lines[key][i+1]['element']

                line_list = [curt_line]

                while curt_line.voverlap(next_line) > 0:
                    line_list.append(next_line)
                    i += 1
                    if i + 1 >= lines_num:
                        break
                    next_line = self.text_lines[key][i+1]['element']

                if len(line_list) > 1:
                    line_list = sorted(line_list, key=attrgetter('x0'))
                    merged_line = line_list[0]
                    x0, y0, x1, y1 = merged_line.x0, merged_line.y0,\
                        merged_line.x1, merged_line.y1
                    for line_part in line_list[1:]:  # merged_line first
                        x0 = min(line_part.x0, x0)
                        y0 = min(line_part.y0, y0)
                        x1 = max(line_part.x1, x1)
                        y1 = max(line_part.y1, x1)
                        bbox = [x0, y0, x1, y1]
                        merged_line._objs += line_part._objs
                    merged_line.set_bbox(bbox)
                else:
                    merged_line = curt_line

                merged_line_dict = {'block_id': curt_iblock,
                                    'element': merged_line}

                merged_lines[key].append(merged_line_dict)
                i += 1

        return merged_lines

    def merge_blocks(self, text_lines):
        """
        Merge lines that have the same block id
        Args: text_line: list of text lines with block id
        Return: blocks composed of several text lines and new block id
        """
        merged_blocks = {}
        for col in ['left', 'right']:
            '''first_line = text_lines[col][0]
            merged_blocks[col] = {first_line['block_id']:[first_line['element']]}'''
            merged_blocks[col] = {}
            for line_dict in text_lines[col]:
                iblock = line_dict['block_id']
                text_line = line_dict['element']
                if iblock in merged_blocks[col].keys():
                    merged_blocks[col][iblock].append(text_line)
                else:
                    merged_blocks[col][iblock] = [text_line]

        return merged_blocks

    # =============================================
    # functions for table extraction and formatting
    # =============================================

    def get_tables(self):

        # 1. Extract left and right tables
        tables = {'left': [], 'right': []}
        for raw_table in self.raw_tables:
            headers = raw_table['content'][0]
            for ih, header in enumerate(headers):
                if header is None:
                    headers[ih] = ''
            partitions = [', '] * (len(headers)-1) + ['。']
            if len(raw_table['content']) > 1 and \
               len(raw_table['content'][0]) > 1:
                for row in raw_table['content'][1:]:
                    for i, item in enumerate(row):
                        if item is None:
                            row[i] = '(' + headers[i] + ') ' + \
                                ' ' + partitions[i]
                        else:
                            row[i] = '(' + headers[i] + ') ' + \
                                item + partitions[i]

            if raw_table['x1'] >= self.middle:
                # important: using the right x position for table!
                tables['right'].append((raw_table['y0'], raw_table))
            else:
                tables['left'].append((raw_table['y0'], raw_table))

        tables['left'] = sorted(tables['left'], key=lambda x: x[0])
        tables['right'] = sorted(tables['right'], key=lambda x: x[0])

        #  2. Extract buttons
        # buttons = self.detect_table_buttons(tables)
        return tables

    # ============================================
    # functions for images and drawings extraction
    # ============================================

    def get_images_drawings(self, save_folder, page_img):
        save_folder = os.path.join(save_folder, 'images')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        images = {'left': [], 'right': []}
        # check images
        # extract and save images
        for imgbox in self.page.images:
            bbox = (imgbox['x0'], imgbox['y0'],
                    imgbox['x1'], imgbox['y1'])
            if isvisible(bbox, self.pbox, margins=self.margins):
                bbox = miner2img(bbox, self.pbox)
                saved_file = save_image(
                    imgbox, page_number=self.page.page_number,
                    images_folder=save_folder)
            if saved_file:
                img_info = {'pageid': self.pid,
                            'bbox': bbox,
                            'path': '<img src="' +
                            os.path.join(save_folder, saved_file)+'" />'}
            else:
                img_info = {'pageid': self.pid,
                            'bbox': bbox,
                            'path': '<img src= NOT SAVED/>'}

            if (bbox[0] >= self.middle):
                images['right'].append(img_info)
            else:
                images['left'].append(img_info)

        # extract and save drawings
        if not self.buttons_detected:
            self.buttons_detected = self.detect_buttons()

        drw_id = 0
        for key in ['left', 'right']:
            for (i, drawing) in enumerate(self.drawings[key]):

                if not self.buttons[i]:
                    bbox = drawing.bbox
                    saved_file = '%d_Draw%05d.png' % (self.pid, drw_id)
                    drw_id += 1
                    file_name = os.path.join(save_folder, saved_file)
                    bbox_crop = page_img.crop(bbox)
                    bbox_crop.save(file_name, quality=100)
                    img_info = {'pageid': self.pid,
                                'bbox': drawing.bbox,
                                'path': '<img src="' + file_name + '" />'}
                    images[key].append(img_info)

        return images['left'] + images['right']
