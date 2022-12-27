import fitz
#pip3 install pymupdf
### READ IN PDF
doc = fitz.open("sample.pdf")

for page in doc:
    ### SEARCH
    text = "Simple PDF"
    text_instances = page.search_for(text)

    ### HIGHLIGHT
    for inst in text_instances:
        highlight = page.add_highlight_annot(inst)
        highlight.update()


### OUTPUT
doc.save("output.pdf", garbage=4, deflate=True, clean=True)