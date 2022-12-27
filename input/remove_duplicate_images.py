# https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
# https://ourcodeworld.com/articles/read/1006/how-to-determine-whether-2-images-are-equal-or-not-with-the-perceptual-hash-in-python
# average_hash, dhash, phash, whash

# pip install imutils

from imutils import paths
import argparse
import time
import sys
import os

import imagehash # pip install imagehash
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
    help="dataset of images to search through (i.e., the haytack)")
ap.add_argument("-n", "--needles", required=False,
    help="set of images we are searching for (i.e., needles)")
args = vars(ap.parse_args())

# grab the paths to both the haystack and needle images 
print("[INFO] computing hashes for haystack...")
haystackPaths = list(paths.list_images(args["haystack"]))

# remove the `\` character from any filenames containing a space
# (assuming you're executing the code on a Unix machine)
if sys.platform != "win32":
    haystackPaths = [p.replace("\\", "") for p in haystackPaths]

# grab the base subdirectories for the needle paths, initialize the
# dictionary that will map the image hash to corresponding image,
# hashes, then start the timer
haystack = {}
start = time.time()

HDBatmanHash = imagehash.whash(Image.open('pokemon/mewtwo/00000090.jpg'))
print('Batman HD Picture: ' + str(HDBatmanHash))

# Create the Hash Object of the second image
SDBatmanHash = imagehash.whash(Image.open('pokemon/mewtwo/00000092.jpg'))
print('Batman HD Picture: ' + str(SDBatmanHash))

# Compare hashes to determine whether the pictures are the same or not
if(HDBatmanHash == SDBatmanHash):
    print("The pictures are perceptually the same !")
else:
    print("The pictures are different, distance: " + str(HDBatmanHash - SDBatmanHash))

# loop over the haystack paths
for p in haystackPaths:
    # load the image from disk
    image = Image.open(p)

    # if the image is None then we could not load it from disk (so
    # skip it)
    if image is None:
        continue

    # convert the image to grayscale and compute the hash
    imageHash = str(imagehash.whash(image))

    # update the haystack dictionary
    l = haystack.get(imageHash, [])
    l.append(p)
    haystack[imageHash] = l
print(haystack)
# Check duplicates in haystack
for h in haystack:
    if len(haystack[h]) > 1:
        print("[DUPLICATE] Image in %s: %s" % (args["haystack"], haystack[h]))
        stats = []
        for p in haystack[h]:
            size = os.stat(p).st_size
            stats[p] = size
        
        smallest = ""
        for i, p in enumerate(stats):
            if i == 0:
                smallest = p
                continue
            if stats[smallest] <= stats[p]:
                os.remove(p)
                continue
            if stats[smallest] > stats[p]:
                os.remove(smallest)
                smallest = p
                continue
        # count = 0
        # for p in haystack[h]:
        #     if count == 0:
        #         count+=1
        #         continue
        #     print("[REMOVE] %s" % (p))
        #     os.remove(p)

if args["needles"] == None:
    exit()

needlePaths = list(paths.list_images(args["needles"]))
BASE_PATHS = set([p.split(os.path.sep)[-2] for p in needlePaths])
if sys.platform != "win32":
    needlePaths = [p.replace("\\", "") for p in needlePaths]

# show timing for hashing haystack images, then start computing the
# hashes for needle images
print("[INFO] processed {} images in {:.2f} seconds".format(
    len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")

# loop over the needle paths
for p in needlePaths:
    # load the image from disk
    image = Image.open(p)

    # if the image is None then we could not load it from disk (so
    # skip it)
    if image is None:
        continue

    # convert the image to grayscale and compute the hash
    imageHash = str(imagehash.whash(image))

    # grab all image paths that match the hash
    matchedPaths = haystack.get(imageHash, [])
    
    if len(matchedPaths) > 0:
        print("[DUPLICATE] Image in %s: %s" % (args["needles"], p))
        #print("[REMOVE] %s" % (p))
        #os.remove(p)

    # loop over all matched paths
    for matchedPath in matchedPaths:
        # extract the subdirectory from the image path
        b = p.split(os.path.sep)[-2]
        # if the subdirectory exists in the base path for the needle
        # images, remove it
        if b in BASE_PATHS:
            BASE_PATHS.remove(b)

# display directories to check
print("[INFO] check the following directories...")

# loop over each subdirectory and display it
for b in BASE_PATHS:
    print("[INFO] {}".format(b))