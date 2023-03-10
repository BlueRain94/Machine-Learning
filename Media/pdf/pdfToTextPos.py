from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfdevice import PDFDevice
import pdfminer

# Open a PDF file.
fp = open('example/azota/multiple-choices.pdf', 'rb')

# Create a PDF parser object associated with the file object.
parser = PDFParser(fp)

# Create a PDF document object that stores the document structure.
# Password for initialization as 2nd parameter
document = PDFDocument(parser)

# Check if the document allows text extraction. If not, abort.
if not document.is_extractable:
    raise PDFTextExtractionNotAllowed

# Create a PDF resource manager object that stores shared resources.
rsrcmgr = PDFResourceManager()

# Create a PDF device object.
device = PDFDevice(rsrcmgr)

# BEGIN LAYOUT ANALYSIS
# Set parameters for analysis.
laparams = LAParams()

# Create a PDF page aggregator object.
device = PDFPageAggregator(rsrcmgr, laparams=laparams)

# Create a PDF interpreter object.
interpreter = PDFPageInterpreter(rsrcmgr, device)

pages = PDFPage.get_pages(fp)

def parse_obj(lt_objs):
    # loop over the object list
    for obj in lt_objs:

        # if it's a textbox, print text and location
        if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
            print("%2d, %2d, %2d, %2d, %s" % (obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3], obj.get_text()))

        # if it's a container, recurse
        elif isinstance(obj, pdfminer.layout.LTFigure):
            parse_obj(obj._objs)

# loop over all pages in the document
for page in PDFPage.create_pages(document):

    # read the page into a layout object
    interpreter.process_page(page)
    layout = device.get_result()

    # extract text from this object
    parse_obj(layout._objs)

fp.close()
#outputfile.close()