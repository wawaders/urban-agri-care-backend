import io
import random

from PIL import Image, ImageDraw

random.seed(7)


def make_image(path, leaf_color, lesions):
    image = Image.new("RGB", (512, 512), (236, 236, 230))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 60, 472, 470), fill=leaf_color)

    for _ in range(lesions):
        x = random.randint(120, 380)
        y = random.randint(140, 390)
        size = random.randint(18, 45)
        draw.ellipse((x, y, x + size, y + size), fill=(124, 74, 30))

    image.save(path)
    print("wrote", path)


make_image("test_healthy.png", (58, 133, 56), 0)
make_image("test_blight.png", (58, 133, 56), 40)

# Non-leaf image: mostly grey, no green tissue.
Image.new("RGB", (256, 256), (128, 128, 128)).save("test_no_leaf.png")

# Valid extension/content type but garbage bytes.
with open("test_corrupt.png", "wb") as file:
    file.write(b"this is definitely not a png")

# Empty file.
open("test_empty.png", "wb").close()

with io.open("checksum.txt", "w", encoding="utf-8") as file:
    file.write("ok")
