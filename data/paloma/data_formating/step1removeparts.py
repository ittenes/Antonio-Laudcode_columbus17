

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
        print (namefile)
        img = Image.open("imagesok/{0}.jpg".format(namefile))
        withimg, heightimg = img.size
        withpercent = e[1]/withimg
        heigthpercent = e[2]/heightimg
        filenew = open("labelsok/{0}.txt".format(namefile), "a")
        filenew.write("1 {0} {1} {2} {2}\n".format(withpercent, heigthpercent,float(10)/float(heightimg)))
        filenew.close()
        print (e)

def cleanproyect(newpath):
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
    if not os.path.exists(newpath):
        os.makedirs(newpath)


def moveAndRename(fileList, selected):
    for line in fileList:
        idb, path = line.split(" ")
        if int(idb) in selected:
            shutil.copyfile("images/" + path[0:len(path)-1], "imagesok/"+idb+".jpg")







cleanproyect("./labelsok")
cleanproyect("./imagesok")
allPartsOfBirds = open("part_locs.txt", "r")
eyePartsOfBirds = filterOnlyEyeParts(allPartsOfBirds)
filebirds= open("images.txt","r")
moveAndRename(filebirds,selected)
createListInSingleTextFilePerBirdId(eyePartsOfBirds)
