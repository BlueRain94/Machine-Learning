# Import Libraries
from pdf2docx import Converter
from pdf2docx import parse

pdf_file = 'math_exam.pdf'
docx_file = 'math_exam.docx'

# # Converting PDF to Docx
# cv_obj = Converter(pdf_file)
# cv_obj.convert(docx_file)
# cv_obj.close()

parse(pdf_file, docx_file)


