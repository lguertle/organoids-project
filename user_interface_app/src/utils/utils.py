from PIL import Image

def resize_image(input_path, output_path, size):
    image = Image.open(input_path)
    image = image.resize(size, Image.Resampling.LANCZOS)
    image.save(output_path)