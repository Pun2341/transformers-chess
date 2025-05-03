import struct
import mmap
import zstandard as zstd
import os


class BagReader:
    def __init__(self, filename):
        self.filename = filename
        self._open_file()
        self._process = lambda x: zstd.decompress(x) if x else x

    def _open_file(self):
        fd = os.open(self.filename, os.O_RDONLY)
        self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        file_size = self._records.size()

        self._records.seek(-8, os.SEEK_END)
        (self.index_start,) = struct.unpack('<Q', self._records.read(8))
        self.num_records = (file_size - self.index_start - 8) // 8

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        if not 0 <= idx < self.num_records:
            raise IndexError('BagReader index out of range')

        limits_offset = self.index_start + idx * 8

        if idx > 0:
            self._records.seek(limits_offset - 8)
            start, end = struct.unpack('<2q', self._records.read(16))
        else:
            self._records.seek(limits_offset)
            (end,) = struct.unpack('<q', self._records.read(8))
            start = 0

        return self._records[start:end]

    def close(self):
        self._records.close()
        self.file.close()
