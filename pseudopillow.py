def read_bmp(fn):
    with open(fn, 'rb') as f:
        data = f.read()
        offset = data[10] + (data[11] << 8) + (data[12] << 16) + (data[13] << 24)

        width = data[18] + (data[19] << 8) + (data[20] << 16) + (data[21] << 24)
        height = data[22] + (data[23] << 8) + (data[24] << 16) + (data[25] << 24)

        # Bits per pixel (byte 28-29)
        bpp = data[28] + (data[29] << 8)
        assert bpp == 8

        palette_size = 1024
        pixel_array = data[offset:offset + width * height]

        pixels = []
        for row in range(height):
            start = (height - 1 - row) * width
            end = start + width
            row_data = [val / 255 for val in pixel_array[start:end]]  # normalize 0-1
            pixels.append(row_data)

        return pixels
