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
	help="search query to search Contextual web Image API for")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())

MAX_RESULTS = 250 # Get first 250 results
pageSize = 50 # Group by 50 results per page

# set the endpoint API URL
URL = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/Search/ImageSearchAPI"

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
	"X-RapidAPI-Key": "adc2cf65a3msh953549090ba4a42p12fb50jsn82e94fe81b35",
	"X-RapidAPI-Host": "contextualwebsearch-websearch-v1.p.rapidapi.com"
}
params = {"q": term, "pageNumber": "1", "pageSize": pageSize}

# make the search
print("[INFO] searching Contextual web API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# grab the results from the search, including the total number of
# estimated results returned by the Contextual web API
results = search.json()
estNumResults = min(results["totalCount"], MAX_RESULTS)
numberOfTotalPages=int(estNumResults/pageSize)

print("[INFO] {} total results for '{}'".format(estNumResults,term))
# initialize the total number of images downloaded thus far
total = 0

# loop over the estimated number of results in `pageSize` groups
for pageNumber in range(1, numberOfTotalPages, 1):
	# update the search parameters using the current pageNumber, then
	# make the request to fetch the results
	print("[INFO] making request for group {}-{} of {}...".format(
		pageNumber, pageNumber + pageSize, estNumResults))
	params["pageNumber"] = pageNumber
	search = requests.get(URL, headers=headers, params=params)
	search.raise_for_status()
	results = search.json()
	print("[INFO] saving images for group {}-{} of {}...".format(
		pageNumber, pageNumber + pageSize, estNumResults))

	# loop over the results
	for v in results["value"]:
		# try to download the image
		try:
			# make a request to download the image
			print("[INFO] fetching: {}".format(v["url"]))
			r = requests.get(v["url"], timeout=30)

			# build the path to the output image
			content_type = r.headers['content-type']
			ext = mimetypes.guess_extension(content_type)
			if ext = "None":
                ext = ".jpg"
			
			#ext = v["url"][v["url"].rfind("."):]
			p = os.path.sep.join([args["output"], "{}{}".format(
				str(total).zfill(8), ext)])
				
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
				print("[INFO] skipping: {}".format(v["url"]))
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