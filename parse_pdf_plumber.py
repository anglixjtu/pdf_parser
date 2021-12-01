import argparse
import os

from src.parser.page_parser import PageParser
import pdfplumber
from pdf2image import convert_from_path
import csv


def parse_args():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, dest='file',
                        default='data/manual/NSZN-Z68T.pdf',
                        help='the path to which pdf file to parse')
    parser.add_argument('--save_folder', type=str, dest='save_folder',
                        default='output/manuals',
                        help='which target folder to save outputs')
    parser.add_argument('--begin_pageid', type=int, default=68,
                        help='parse from page #', dest='begin_pageid')
    parser.add_argument('--end_pageid', type=int, default=-1,
                        help='parse ending at page #: -1 for the last page of pdf;'
                        '0 for begin_pageid+1', dest='end_pageid')
    parser.add_argument('-viz_debug', action='store_true', dest='viz_debug',
                        help='trigger visual debug or not')
    parser.add_argument('-get_imgs', action='store_true', dest='get_imgs',
                        help='extract images or not')

    args = parser.parse_args()
    if args.end_pageid == 0:
        args.end_pageid = args.begin_pageid+1
    return args


def main():
    args = parse_args()

    page_elements = {'text': [],
                     'image': []}

    pdf = pdfplumber.open(args.file, laparams={"line_overlap": 0.7})
    if args.end_pageid == -1:
        args.end_pageid = len(pdf.pages)

    # loop for each page
    '''sp_codes = ['\u3000',  '\uf020', '\uf02a', '\uf06c', '\uf06e', '\uf074', '\uf075', '\uf0ae',
                '\uf0b7', '\uf0bd', '\uf0be', '\uf0c6',
                '\uf0e1', '\uf0e2', '\uf0e3', '\uf0e4']
    checked = [False] * len(sp_codes)'''
    for pageid in range(args.begin_pageid, args.end_pageid, 1):
        page = pdf.pages[pageid]
        extractor = PageParser(page)
        text_lines = extractor.get_text()
        tables = extractor.get_tables()
        page_sequence = extractor.merge_page(text_lines, tables)
        # p73,table, 162
        '''for i, sp_code in enumerate(sp_codes):
            if sp_code in page_sequence and not checked[i]:
                print(page_sequence)
                print(page.page_number)
                print(sp_code)
                checked[i] = True'''

        page_elements['text'].append({'pageid': page.page_number,
                                      'sequence': page_sequence})

        # for images
        if args.get_imgs:
            page_img = convert_from_path(args.file, first_page=pageid+1,
                                         last_page=pageid+1, use_cropbox=True,
                                         size=(extractor.pw, extractor.ph))
            page_images = extractor.get_images_drawings(args.save_folder,
                                                        page_img[0])

            page_elements['image'] += page_images

    print('Parsing done!\nWriting to csv...')

    # write to csv file
    pdf_name = os.path.split(args.file)[1].split('.')[0]
    text_file = os.path.join(args.save_folder, pdf_name+'.csv')
    with open(text_file, 'w', newline='', encoding="utf_8") as csvfile:
        fieldnames = ['page', 'sequence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for page in page_elements['text']:
            row = {'page': page['pageid'], 'sequence': page['sequence']}
            writer.writerow(row)

    if args.get_imgs:
        image_file = os.path.join(args.save_folder, pdf_name+'_imgs.csv')
        with open(image_file, 'w', newline='', encoding="utf_8") as csvfile:
            fieldnames = ['page', 'bbox', 'path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for page in page_elements['image']:
                row = {'page': page['pageid'], 'bbox': page['bbox'],
                       'path': page['path']}
                writer.writerow(row)
    print('Finished!')


if __name__ == '__main__':
    main()
