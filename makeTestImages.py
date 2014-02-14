#!/bin/env python26
#-------------------------------------------------------------------------------

try:
    from PIL import Image, ImageDraw
except ImportError:
    import Image, ImageDraw # linux

from path import path as Path

width = 1024
height = width
size = (width,height)

im = Image.new("RGB",size,0)
draw = ImageDraw.Draw(im)

draw.polygon( [
    (0,height-1),
    (0,0),
    (width-1,0)
    ], fill=(128,128,128))

draw.ellipse((6, 6, 70, 70), fill=(255,0,0))
draw.ellipse((6+0.833*width, 6+0.833*height, 70+0.833*width, 70+0.833*height), fill=(255,0,0))

draw.polygon( [
    (0.333*width,0.666*height-1),
    (0.333*width,0.333*height),
    (0.666*width-1,0.333*height)
    ], fill=(255,255,0))
im.save("testImageA.png")


im = Image.new("RGB",size,0)
draw = ImageDraw.Draw(im)

draw.polygon( [
    (0,height-1),
    (0,0),
    (width-1,0),
    ], fill=(128,128,128))

draw.ellipse((6, 6, 70, 70), fill=(0,255,0))
draw.ellipse((6+0.833*width, 6+0.833*height, 70+0.833*width, 70+0.833*height), fill=(0,255,0))
             
draw.polygon( [
    (0.333*width,0.666*height-1),
    (0.333*width,0.333*height),
    (0.666*width-1,0.333*height),
    (0.666*width-1,0.666*height-1),
    ], fill=(255,255,0))
im.save("testImageB.png")
