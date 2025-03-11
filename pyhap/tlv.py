"""Encodes and decodes Tag-Length-Value (tlv8) data."""
import struct
from typing import Any

from pyhap import util


def encode(*args, to_base64=False):
    """Encode the given byte args in TLV format.

    :param args: Even-number, variable length positional arguments repeating a tag
        followed by a value.
    :type args: ``bytes``

    :param toBase64: Whether to encode the resuting TLV byte sequence to a base64 str.
    :type toBase64: ``bool``

    :return: The args in TLV format
    :rtype: ``bytes`` if ``toBase64`` is False and ``str`` otherwise.
    """
    arg_len = len(args)
    if arg_len % 2 != 0:
        raise ValueError(f"Even number of args expected ({arg_len} given)")

    pieces = []
    for x in range(0, len(args), 2):
        tag = args[x]
        data = args[x + 1]
        total_length = len(data)
        if len(data) <= 255:
            encoded = tag + struct.pack("B", total_length) + data
        else:
            encoded = b""
            for y in range(total_length // 255):
                encoded = encoded + tag + b"\xFF" + data[y * 255 : (y + 1) * 255]
            remaining = total_length % 255
            encoded = encoded + tag + struct.pack("B", remaining) + data[-remaining:]

        pieces.append(encoded)

    result = b"".join(pieces)

    return util.to_base64_str(result) if to_base64 else result


def decode(data: bytes, from_base64: bool = False) -> dict[bytes, Any]:
    """Decode the given TLV-encoded ``data`` to a ``dict``.

    :param from_base64: Whether the given ``data`` should be base64 decoded first.
    :type from_base64: ``bool``

    :return: A ``dict`` containing the tags as keys and the values as values.
    :rtype: ``dict``
    """
    if from_base64:
        data = util.base64_to_bytes(data)

    objects = {}
    current = 0
    while current < len(data):
        # The following hack is because bytes[x] is an int
        # and we want to keep the tag as a byte.
        tag = data[current : current + 1]
        length = data[current + 1]
        value = data[current + 2 : current + 2 + length]
        if tag in objects:
            objects[tag] = objects[tag] + value
        else:
            objects[tag] = value

        current = current + 2 + length

    return objects

####
def read_uint64_le(buffer: bytes, offset: int = 0) -> int:
    low = struct.unpack_from("<I", buffer, offset)[0]
    high = struct.unpack_from("<I", buffer, offset + 4)[0]
    return (high << 32) | low

def read_uint64_be(buffer: bytes, offset: int = 0) -> int:
    return read_uint64_le(buffer, offset)

def write_uint32(value: int) -> bytes:
    return struct.pack("<I", value)

def read_uint32(buffer: bytes) -> int:
    return struct.unpack("<I", buffer)[0]

def write_float32_le(value: float) -> bytes:
    return struct.pack("<f", value)

def write_uint16(value: int) -> bytes:
    return struct.pack("<H", value)

def read_uint16(buffer: bytes) -> int:
    return struct.unpack("<H", buffer)[0]

def read_uint32_le(buffer, offset):
    # Extract the 4 bytes from the buffer starting at the specified offset
    bytes_slice = buffer[offset:offset + 4]

    # Convert the bytes to an unsigned, little-endian 32-bit integer
    uint32_value = (bytes_slice[0] << 0) | (bytes_slice[1] << 8) | (bytes_slice[2] << 16) | (bytes_slice[3] << 24)

    return uint32_value


def read_float_le(buffer, offset):
    # Extract the 4 bytes from the buffer starting at the specified offset
    bytes_slice = buffer[offset:offset + 4]

    # Convert the bytes to a little-endian float using struct.unpack
    float_value = struct.unpack('<f', bytes_slice)[0]

    return float_value

# def read_variable_uint_le(buffer: bytes, offset: int = 0) -> int:
#     switch = {
#         1: struct.unpack_from("<B", buffer, offset)[0],
#         2: struct.unpack_from("<H", buffer, offset)[0],
#         4: struct.unpack_from("<I", buffer, offset)[0],
#         8: read_uint64_le(buffer, offset),
#     }
#     length = len(buffer)
#     if length in switch:
#         return switch[length]
#     else:
#         raise ValueError("Can't read uint LE with length " + str(length))

def read_variable_uint_le(buffer: bytes, offset: int = 0) -> int:
    result = 0
    shift = 0
    while True:
        # Read one byte at a time
        byte = struct.unpack_from("<B", buffer, offset)[0]
        result |= (byte & 0x7F) << shift
        offset += 1
        shift += 7
        if not (byte & 0x80):  # Check if the MSB is set
            break
    return result

def write_variable_uint_le(number: int) -> bytes:
    if number <= 255:
        return struct.pack("<B", number)
    if number <= 65535:
        return struct.pack("<H", number)
    if number <= 4294967295:
        return struct.pack("<I", number)
    return struct.pack("<Q", number)

# Constants
EMPTY_TLV_TYPE = 0x00

def decode_with_lists(buffer: bytes) -> dict[int, bytes | list[bytes]]:
    result = {}
    left_bytes = len(buffer)
    read_index = 0

    last_type = -1
    last_length = -1
    last_item_was_delimiter = False

    while left_bytes > 0:
        tag = buffer[read_index]
        length = buffer[read_index + 1]
        read_index += 2
        left_bytes -= 2

        data = buffer[read_index:read_index + length]
        read_index += length
        left_bytes -= length

        if tag == 0x00 and length == 0x00:
            last_item_was_delimiter = True
            continue

        existing = result.get(tag)
        if existing is not None:
            if last_item_was_delimiter and last_type == tag:
                if isinstance(existing, list):
                    existing.append(data)
                else:
                    result[tag] = [existing, data]
            elif last_type == tag and last_length == 0xFF:
                if isinstance(existing, list):
                    # append to the last data blob in the array
                    last = existing[-1]
                    existing[-1] = last + data
                else:
                    result[tag] = existing + data
            else:
                raise ValueError(f"Found duplicated tlv entry with type {tag} and length {length}")
        else:
            result[tag] = data

        last_type = tag
        last_length = length
        last_item_was_delimiter = False

    return result
