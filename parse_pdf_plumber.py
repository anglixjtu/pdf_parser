import argparse
import os

from src.parser.page_parser import PageParser
from src.parser.specical_char_handler import (define_map,
                                              write_sp_codes,
                                              load_sp_codes,
                                              replace_sp_codes)
import pdfplumber
from pdf2image import convert_from_path
import csv


def parse_args():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, dest='file',
                        default='data/manual/open_data/harrier_navi_201706.pdf',
                        help='the path to which pdf file to parse')
    parser.add_argument('--save_folder', type=str, dest='save_folder',
                        default='data/manual/open_data',
                        help='which target folder to save outputs')
    parser.add_argument('--begin_pageid', type=int, default=13,
                        help='parse from page #', dest='begin_pageid')
    parser.add_argument('--end_pageid', type=int, default=30, #563
                        help='parse ending at page #: -1 for the'
                        'last page of pdf;'
                        '0 for begin_pageid+1', dest='end_pageid')
    parser.add_argument('--pid_h', type=int, default=30,
                        help='the height of page id show in the pdf',
                        dest='pid_h')
    parser.add_argument('--out_spcode', type=str, dest='out_spcode',
                        default=None,
                        help='The output file stores the mapping of special codes')
    parser.add_argument('--in_spcode', type=str, dest='in_spcode',
                        default='D:/ARIH/Test-Cases-Generation/workspace/code/'
                                'test_cases_generation/config/harrier_navi_201706_spcodes.json',
                        help='The input file stores the mapping of special codes')
    parser.add_argument('--check_spcode', action='store_true', dest='check_spcode',
                        help='check the loaded/stored spcodes or not')
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
    pdf_name = os.path.split(args.file)[1].split('.')[0]

    page_elements = {'text': [],
                     'image': []}

    pdf = pdfplumber.open(args.file, laparams={"line_overlap": 0.7})
    if args.end_pageid == -1:
        args.end_pageid = len(pdf.pages)

    # loop for each page
    '''sp_codes = ['\u200b',  '\uf0e2', '〓', '\uf06e', '◯', '◀', '■', '〔',
                '〕', '〈', '×', '>',
                '<', '〉', '○', '°']
    checked = [False] * len(sp_codes)'''
    if args.in_spcode is None:
        sp_codes = {}
    else:
        in_sp_file = args.in_spcode
        sp_codes = load_sp_codes(in_sp_file)

    for pageid in range(args.begin_pageid, args.end_pageid, 1):
        page = pdf.pages[pageid]
        extractor = PageParser(page, pid_h=args.pid_h, sp_codes=sp_codes, pageid=pageid)
        text_blocks = extractor.get_text()
        tables = extractor.get_tables()
        page_sequences = extractor.get_sequences(text_blocks, tables)
        # p73,table, 162
        '''for i, sp_code in enumerate(sp_codes):
            if sp_code in page_sequence and not checked[i]:
                print(page_sequence)
                print(page.page_number)
                print(sp_code)
                checked[i] = True'''

        for seq in page_sequences:
            page_elements['text'].append({'pageid': extractor.pid,
                                          'sequence': seq})
            # page_elements['text'].append({'pageid': page.page_number,
            #                               'sequence': seq})

        # for images
        if args.get_imgs:
            page_img = convert_from_path(args.file, first_page=pageid+1,
                                         last_page=pageid+1, use_cropbox=True,
                                         size=(extractor.pw, extractor.ph))
            page_images = extractor.get_images_drawings(args.save_folder,
                                                        page_img[0])

            page_elements['image'] += page_images

    # handling special codes in the extracted string
    ## load and store special codes mappings
    '''if (args.in_spcode is None) or \
        (args.in_spcode is not None and args.check_spcode):'''
    if args.check_spcode:
        sp_codes = define_map(sp_codes, args.file, page_size=(extractor.pw, extractor.ph))
        if args.out_spcode is None:
            out_file = os.path.join(args.save_folder, pdf_name+'_spcodes.json')
        else:
            out_file = os.path.join(args.save_folder, args.out_spcode+'_spcodes.json')
        write_sp_codes(sp_codes, out_file)
    ## replace special codes with the mapping
    page_elements['text'] = replace_sp_codes(sp_codes, page_elements['text'])
    
    print('Parsing done!\nWriting to csv...')

    # write to csv file
    
    text_file = os.path.join(args.save_folder, pdf_name+'.csv')
    with open(text_file, 'w', newline='', encoding="utf_8") as csvfile:
        fieldnames = ['page', 'sequence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for page in page_elements['text']:
            if len(page['sequence']) == 0:
                continue
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
