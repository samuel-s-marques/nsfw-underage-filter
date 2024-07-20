import json
import base64
import binascii

def parse_png(bytes_data):
    if not is_valid_png(bytes_data):
        return None

    decoded_png = get_text_chunk(bytes_data)

    if decoded_png is None:
        return None

    return json.loads(decoded_png)


def is_valid_png(bytes_data):
    if len(bytes_data) < 8:
        return False

    png_signature = b"\x89PNG\r\n\x1a\n"
    return bytes_data.startswith(png_signature)


def get_text_chunk(bytes_data):
    offset = 8
    base64_data = None

    while offset < len(bytes_data):
        if offset + 8 > len(bytes_data):
            break

        length = int.from_bytes(bytes_data[offset : offset + 4], byteorder="big")
        offset += 4

        chunk_type = bytes_data[offset : offset + 4].decode("utf-8")
        offset += 4

        if chunk_type == "tEXt":
            null_separator_index = bytes_data.index(0, offset)
            if null_separator_index == -1:
                break

            try:
                text = bytes_data[null_separator_index + 1 : offset + length].decode(
                    "utf-8"
                )
                base64_data = base64.b64decode(text).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError):
                return None

        offset += length + 4

    return base64_data
