from io import BytesIO


def to_bytesio(image):
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    return img_io
