def miner2img(bbox, page_bbox):
    """Convert the coordinates in pdfminer to coordinates in image.
    In pdfminer, y is increasing from the bottom to the top.
    The visible pdf region is start from (x0, y0).
    In image, y is increasing from top to the bottom.
    """
    ltx, rby, rbx, lty = bbox
    x0, y0, x1, y1 = page_bbox
    H = y1 - y0
    ltx, rby, rbx, lty, H, x0, y0 =\
        float(ltx), float(rby), float(rbx), float(lty),\
        float(H), float(x0), float(y0)
    return ltx-x0, H - (lty-y0), rbx-x0, H-(rby-y0)


def isvisible(obj_bbox, pos_page, margins=[0, 0, 0, 0]):
    ltx, lty, rbx, rby = obj_bbox
    x0, y0, x1, y1 = pos_page
    marginl, marginr, margint, marginb = margins
    return ltx >= x0 + marginl and\
        lty >= y0 + margint and\
        rbx <= x1 - marginr and\
        rby <= y1 - marginb
