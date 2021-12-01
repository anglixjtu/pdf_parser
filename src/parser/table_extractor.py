

def get_raw_tables(page):
    data = []

    x0, y0, x1, y1 = page.cropbox[0], page.cropbox[1], page.cropbox[2], page.cropbox[3]
    width = x1.real - x0.real
    height = y1.real - y0.real

    tables = page.find_tables()
    table_id = 1
    for itable, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox[0]-x0, table.bbox[1]-y0, \
            table.bbox[2]-x0, table.bbox[3]-y0

        in_table = False
        for ictable, check_table in enumerate(tables):
            if itable == ictable:
                continue
            ctx0, cty0, ctx1, cty1 = check_table.bbox[0]-x0,\
                check_table.bbox[1]-y0, \
                check_table.bbox[2]-x0, check_table.bbox[3]-y0
            if tx0 >= ctx0 and ty0 >= cty0 and tx1 <= ctx1 and ty1 <= cty1:
                in_table = True
                break
        if in_table:
            continue

        if tx0 > 0 and tx0 < width and tx1 > 0 and tx1 < width and \
                ty0 > 0 and ty0 < height and ty1 > 0 and ty1 < height:
            table_content = table.extract()
            all_null = True
            for row in table_content:
                for cell in row:
                    if cell is not None:
                        if len(cell) > 0:
                            all_null = False
                            break
                if not all_null:
                    break
            if (not all_null) and len(table_content) > 1 and\
                    len(table_content[0]) > 1:
                cells = []
                '''if len(table_content[0])*len(table_content) != len(table.cells):
                    debug = True'''
                for irow, row in enumerate(table.rows):
                    for icell, cell in enumerate(row.cells):
                        if cell is None:
                            for icrow, check_row in enumerate(table.rows):
                                if (row.bbox[1] >= check_row.bbox[1]) and\
                                        (row.bbox[3] <= check_row.bbox[3]):
                                    table_content[irow][icell] = \
                                        table_content[icrow][icell]

                                    cell = {}

                                    check_bbox = True
                        # cells.append(
                        #    (cell[0]-x0, cell[1]-y0, cell[2]-x0, cell[3]-y0))
                table_data = {'x0': tx0, 'y0': ty0, 'x1': tx1, 'y1': ty1,
                              'height': ty1-ty0, 'width': tx1-tx0,
                              'content': table_content,
                              'table_id': table_id,
                              'rows': len(table_content),
                              'cols': len(table_content[0])}  # ,
                # 'cells': cells}
                table_id += 1
                data.append(table_data)

    return data


def match_table_text(tables, texts):
    def is_contain(table_bbox, text_bbox):
        tolerance = 2
        if text_bbox[0] >= table_bbox[0]-tolerance and\
            text_bbox[1] >= table_bbox[1]-tolerance and\
                text_bbox[2] <= table_bbox[2]+tolerance and\
        text_bbox[3] <= table_bbox[3]+tolerance:
            return True

    for text in texts:

        for table in tables:
            text_bbox = (text['x0'], text['y0'],
                         text['x1'], text['y1'])
            table_bbox = (table['x0'], table['y0'],
                          table['x1'], table['y1'])
            if not is_contain(table_bbox, text_bbox):
                continue

            for i, cell in enumerate(table['cells']):

                if is_contain(cell, text_bbox):
                    text['table_id'] = table['table_id']
                    text['table_coord'] = i + 1
                    text['cell_bbox'] = cell

    return texts
