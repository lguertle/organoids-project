from PIL import Image
import json
import os
import re

def resize_image(input_path, output_path, size):
    image = Image.open(input_path)
    image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(output_path)

def list_saved_runs(directory="./runs"):
    files = os.listdir(directory)
    return [file for file in files if file.endswith('.json')]

def generate_well_annotations(num_rows, num_columns):
    # Generate letters for rows (A, B, C, ...)
    row_letters = [chr(i) for i in range(65, 65 + num_rows)]  # 65 is ASCII for 'A'

    # Generate well annotations
    wells = [f"{letter}{num + 1}" for letter in row_letters for num in range(num_columns)]

    return wells

def is_not_single_letter_or_number(text):
    # Check if text is a single character (letter or digit)
    if len(text) == 1 and text.isalnum():
        return False
    # Check if text is a number (integer or float)
    if re.match(r"^\d+\.?\d*$", text):
        return False
    return True

def well_annotation_to_number(annotation, num_columns):
    # Extract the row letter and column number from the annotation
    row_letter = annotation[0]  # The first character is the row letter
    col_number = int(annotation[1:])  # The rest of the string is the column number

    # Convert the row letter to a row index (0-based)
    row_index = ord(row_letter.upper()) - ord('A')  # 'A' becomes 0, 'B' becomes 1, etc.

    # Calculate the well number
    well_number = row_index * num_columns + col_number

    return str(well_number)