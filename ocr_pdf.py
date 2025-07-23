from PIL import Image
import pytesseract
import pdf2image
import os

# Path to your PDF and output text file
pdf_path = "hsc26_bangla_1st_paper.pdf"
output_text_path = "output.txt"

# Convert PDF to images (requires poppler, install via: https://github.com/oschwartz10612/poppler-windows)
images = pdf2image.convert_from_path(pdf_path, dpi=300, poppler_path=r"C:\Users\Froggo\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin")

# Extract text from each image
full_text = ""
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang="ben")
    full_text += f"Page {i+1}:\n{text}\n\n"

# Save to text file
with open(output_text_path, "w", encoding="utf-8") as file:
    file.write(full_text)

print(f"Text saved to {output_text_path}")