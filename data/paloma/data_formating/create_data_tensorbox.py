

import shutil as shutil
import os as os
from PIL import Image

selected = [];

def filterOnlyEyeParts(lines):
    alldata = []
    for line in lines:
        arrayOfLineSplittedBySpace = line.split(" ")
        if(arrayOfLineSplittedBySpace[1] == "11" or arrayOfLineSplittedBySpace[1] == "7"):
            if(arrayOfLineSplittedBySpace[4] == '1\n'):
                xBoxed = float(arrayOfLineSplittedBySpace[2])-5
                yBoxed = float(arrayOfLineSplittedBySpace[3])-5
                idBird = float(arrayOfLineSplittedBySpace[0])
                selected.append(int(idBird));
                alldata.append([idBird, xBoxed, yBoxed])
    return alldata


# listdoc()

def createListInSingleTextFilePerBirdId(alldata):
    for e in alldata:
        namefile = int(e[0])
#        print ( "{0} {1} {2}\n".format(e[0],e[1],e[2]))
#        print (e)
        pice = '''
        {{
            "image_path": "cub/{0}.jpg",
            "rects": [
                {{
                    "x1": {1},
                    "x2": {2},
                    "y1": {3},
                    "y2": {4}
                }}
            ]
        }},'''.format(int(e[0]),e[1],e[1]+10,e[2],e[2]+10)
        print(pice)

allPartsOfBirds = open("part_locs.txt", "r")
eyePartsOfBirds = filterOnlyEyeParts(allPartsOfBirds)
createListInSingleTextFilePerBirdId(eyePartsOfBirds)
