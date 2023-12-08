from PIL import Image
import json
import os

def resize_image(input_path, output_path, size):
    image = Image.open(input_path)
    image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(output_path)

def list_saved_runs(directory="."):
    files = os.listdir(directory)
    return [file for file in files if file.endswith('.json')]