# =====================================================
# Define and replace special symbols
#
# =====================================================
import tkinter as tk
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageTk
import json
import re





def define_map(sp_codes, pdf_file, page_size):

    def get_decoded_char(sp_code, entry, root):
        sp_code['char']= entry.get()
        root.destroy()

    

    for code_key in sp_codes.keys():
        root = tk.Tk()
         
        
        pageid = sp_codes[code_key]['pageid']
        block_bbox = sp_codes[code_key]['block_bbox']
        char_bbox = sp_codes[code_key]['char_bbox']
        decoded_char = sp_codes[code_key]['char']

        page_img = convert_from_path(pdf_file, first_page=pageid+1,
                                     last_page=pageid+1, use_cropbox=True,
                                     size=page_size, dpi=1000)
        
        block_bbox = (block_bbox[0]-5,
                      block_bbox[1]-5,
                      block_bbox[2]+5,
                      block_bbox[3]+5)

        char_bbox = (char_bbox[0] - block_bbox[0]-3,
                     char_bbox[1] - block_bbox[1]-3,
                     char_bbox[2] - block_bbox[0]+3,
                     char_bbox[3] - block_bbox[1]+3)

        block_patch = page_img[0].crop(block_bbox)
        draw =ImageDraw.Draw(block_patch)
        draw.rectangle(char_bbox, width=1, outline ="red" )
        
        scale = 3
        bwidth = int(block_patch.size[0]*scale)
        bheight = int(block_patch.size[1]*scale)
        block_patch = block_patch.resize((bwidth, bheight))

        block_patch_tk = ImageTk.PhotoImage(block_patch)

        frame1 = tk.Frame(root)
        frame2 = tk.Frame(root)
        frame3 = tk.Frame(root)
        root.title("Define the code...")

        label_img = tk.Label(frame1, image=block_patch_tk)
        label_img.pack()

        entry = tk.Entry(frame2)
        entry.insert(0, decoded_char)
        entry.pack()

        #Create a Button to validate Entry Widget
        enter_button = tk.Button(frame3, text= "Enter", width= 20,
                                 command=lambda:get_decoded_char(sp_codes[code_key], entry, root))
        enter_button.bind("<Return>", root.destroy)
        enter_button.pack()
 
        frame1.pack(padx=1,pady=1)
        frame2.pack(padx=10,pady=10)
        frame3.pack(padx=20,pady=20)


        root.mainloop()

        debug=True

    return sp_codes


def write_sp_codes(sp_codes, out_file):
    sp_codes = json.dumps(sp_codes, indent=4)  # indent参数是换行和缩进
   
    with open(out_file, 'w+') as fileObject:
        fileObject.write(sp_codes) 


def load_sp_codes(in_file):
    with open(in_file, 'r') as fileObject:
        sp_codes = json.load(fileObject) 
    return sp_codes


def replace_sp_codes(sp_codes, text_items):
    for i, text_item in enumerate(text_items):
        line = text_item['sequence']
        if i == 80:
            debug = True
        for sp_code_key in sp_codes.keys():
            
            if sp_code_key in line:
                target = sp_codes[sp_code_key]['char']
                text_items[i]['sequence']=\
                    text_items[i]['sequence'].replace(sp_code_key, target)

    return text_items

        
        






