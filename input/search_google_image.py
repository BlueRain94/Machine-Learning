# Enable Google Custom Search
# https://console.developers.google.com and create a project.
# https://console.developers.google.com/apis/library/customsearch.googleapis.com and enable "Custom Search API" for your project.
# https://console.developers.google.com/apis/credentials and generate API key credentials for your project.
# https://cse.google.com/cse/all and in the web form where you create/edit your custom search engine 
# enable "Image search" option and for "Sites to search" option select "Search the entire web but emphasize included sites".

# mkdir pokemon/charmander
# python search_contextualweb_api.py --query "charmander" --output pokemon/charmander

# import the necessary packages
from requests import exceptions
import argparse
import requests
import mimetypes
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
    help="search query to search Google Image API for")
ap.add_argument("-o", "--output", required=True,
    help="path to output directory of images")
args = vars(ap.parse_args())

MAX_RESULTS = 100 # Get first 50 results
pageSize = 10 # Results per page, Google only allow 10
API_KEY = "AIzaSyBLG0yw4t1X3bmsLKZ_ZXmX-NCCtjP4a9E"
CSE_ID = "33587360f55774907"

# initialize the total number of images downloaded thus far
total = 0

# set the endpoint API URL
URL = "https://www.googleapis.com/customsearch/v1"

# when attempting to download images from the web both the Python
# programming language and the requests library have a number of
# exceptions that can be thrown so let's build a list of them now
# so we can filter on them
EXCEPTIONS = set([IOError, FileNotFoundError,
    exceptions.RequestException, exceptions.HTTPError,
    exceptions.ConnectionError, exceptions.Timeout])

# store the search term in a convenience variable then set the
# headers and search parameters
term = args["query"]

headers = {
}

params = {"key": API_KEY, "cx": CSE_ID, "q": term, "start" : total, "num": pageSize, "searchType": "image"}
# https://developers.google.com/custom-search/v1/reference/rest/v1/

# make the search
print("[INFO] searching Google API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# grab the results from the search, including the total number of
# estimated results returned by the Google API
results = search.json()
#print(results)

estNumResults = min(int(results["searchInformation"]["totalResults"]), MAX_RESULTS)
#numberOfTotalPages=int(estNumResults/pageSize)
#print(results["searchInformation"]["totalResults"], estNumResults, str(numberOfTotalPages))

print("[INFO] {} total results for '{}'".format(estNumResults,term))

# loop over the estimated number of results in `pageSize` groups
for startIndex in range(0, estNumResults, pageSize):
    # update the search parameters using the current startIndex, then
    # make the request to fetch the results
    print("[INFO] making request for group {}-{} of {}...".format(
        startIndex, startIndex + pageSize, estNumResults))
    params["start"] = startIndex
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(
        startIndex, startIndex + pageSize, estNumResults))

    # loop over the results
    for v in results["items"]:
        # try to download the image
        try:
            # make a request to download the image
            print("[INFO] fetching: {}".format(v["link"]))
            r = requests.get(v["link"], timeout=30)

            # build the path to the output image
            content_type = v["mime"]
            ext = mimetypes.guess_extension(content_type)
            if ext == None:
                ext = ".jpg"

            #ext = v["link"][v["link"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(str(total).zfill(8), ext)])
            print(p)
            
            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()
        # catch any errors that would not unable us to download the
        # image
        except Exception as e:
            # check to see if our exception is in our list of
            # exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["link"]))
                continue

        # try to load the image from disk
        image = cv2.imread(p)
        # if the image is `None` then we could not properly load the
        # image from disk (so it should be ignored)
        if image is None:
            print("[INFO] deleting: {}".format(p))
            os.remove(p)
            continue
        # update the counter
        total += 1