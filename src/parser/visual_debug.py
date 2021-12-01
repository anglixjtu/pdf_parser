import pdfplumber

color_1 = (0, 125, 255)
color_2 = (255, 0, 125)
color_3 = (0, 255, 125)
color_4 = (0, 0, 255)


def visualize_img(file, elements, begin_elemid=0, begin_pageid=0, end_pageid=None):
    from PIL import Image, ImageDraw, ImageFont
    from pdf2image import convert_from_path

    line_width = 1

    pdf = pdfplumber.open(file)
    if end_pageid == None:
        end_pageid = len(pdf.pages)

    for pageid in range(begin_pageid, end_pageid, 1):
        page = pdf.pages[pageid]
        x0, y0, x1, y1 = page.cropbox[0], page.cropbox[1], page.cropbox[2], page.cropbox[3]
        width = x1.real - x0.real
        height = y1.real - y0.real
        page_img = convert_from_path(file, first_page=pageid+1,
                                     last_page=pageid+1, use_cropbox=True, size=(width, height))
        page_img = page_img[0]
        draw = ImageDraw.Draw(page_img)
        element = elements[pageid-begin_elemid]

        fnt = ImageFont.truetype("C:/Windows/Fonts/msgothic.ttc", 7)

        # visualize text and table
        for item in element['text']:
            color = color_1
            bbox = (int(item['x0']), int(item['y0']),
                    int(item['x1']), int(item['y1']))
            draw.rectangle(bbox, outline=color, width=line_width)

            if 'table_id' in item.keys():
                if item['table_id'] > 0:
                    table_info = 'T:%d, (%d, %d)' % (item['table_id'],
                                                     item['table_coord'][0], item['table_coord'][1])
                    draw.text(
                        (int(item['x1']-3), int(item['y0'])-3), table_info, fill=color_4, font=fnt)

        # visualize images
        for item in element['image']:
            color = color_2
            bbox = (int(item['x0']), int(item['y0']),
                    int(item['x1']), int(item['y1']))
            if isinstance(item['content'], list):
                color = color_3
            draw.rectangle(bbox, outline=color, width=line_width)

        # visualize table
        '''color = color_4
        for item in element['table']:
            bbox = (int(item['x0']), int(item['y0']),
                    int(item['x1']), int(item['y1']))
            draw.rectangle(bbox, outline=color, width=line_width)'''

        page_img.show()


def visualize_pdf(file, elements, begin_elemid=0, begin_pageid=0, end_pageid=None, save_file=None):
    import fitz
    from PIL import Image, ImageDraw, ImageFont
    import os

    def draw_bbox(page, bbox, text=None, color=(1, 0, 0.5)):
        page.draw_rect(bbox, color=color)
        if bbox[0] < page.MediaBoxSize[0]/2.-100:
            ant_pt = (bbox[0]-20, bbox[1])
        else:
            ant_pt = (bbox[2]+20, bbox[1])
        if text is not None:
            page.add_text_annot(
                ant_pt, text, icon="Note")

    doc = fitz.open(file)
    if end_pageid == None:
        end_pageid = len(doc)

    for pageid in range(begin_pageid, end_pageid, 1):
        page = doc.load_page(pageid)
        element = elements[pageid-begin_elemid]

        # visualize text and table
        for item in element['text']:
            color = (color_1[0]/255., color_1[1]/255., color_1[2]/255.)
            bbox = (int(item['x0']), int(item['y0']),
                    int(item['x1']), int(item['y1']))
            draw_bbox(page, bbox, color=color)
            rect = (int(item['x1']), int(item['y1']),
                    int(item['x1']+6), int(item['y1']+6))
            page.add_freetext_annot(
                rect, item['content'], fontsize=6, fontname="helv", text_color=color)

            text_color = (color_4[0]/255., color_4[1]/255., color_4[2]/255.)
            if item['table_id'] > 0:
                table_info = 'TABLE_%d: %d' % (item['table_id'],
                                               item['table_coord'])
                rect = (int(item['cell_bbox'][2])-40, int(item['cell_bbox'][1]),
                        int(item['cell_bbox'][2]), int(item['cell_bbox'][1])+10)
                page.add_freetext_annot(
                    rect, table_info, fontsize=6, fontname="helv", text_color=text_color)

        # visualize images
        for item in element['image']:
            color = (color_2[0]/255., color_2[1]/255., color_2[2]/255.)
            bbox = (int(item['x0']), int(item['y0']),
                    int(item['x1']), int(item['y1']))
            if isinstance(item['content'], list):
                color = (color_3[0]/255., color_3[1]/255., color_3[2]/255.)
                draw_bbox(page, bbox, text=item['content'][0], color=color)
            else:
                draw_bbox(page, bbox, color=color)

            text_color = (color_4[0]/255., color_4[1]/255., color_4[2]/255.)
            if item['table_id'] > 0:
                table_info = 'TABLE_%d: %d' % (item['table_id'],
                                               item['table_coord'])
                rect = (int(item['cell_bbox'][2])-40, int(item['cell_bbox'][1]),
                        int(item['cell_bbox'][2]), int(item['cell_bbox'][1])+10)
                page.add_freetext_annot(
                    rect, table_info, fontsize=6, fontname="helv", text_color=text_color)

        if save_file:
            doc.save(save_file)

        debug = True
