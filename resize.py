from PIL import Image
import os, sys

path = "./data/images/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((640,480))
            imResize.save(f + ' r.jpg', 'JPEG', quality=90)

resize()