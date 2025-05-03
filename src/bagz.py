import struct
import mmap
import zstandard as zstd


class BagReader:
    def __init__(self, filename):
        self.filename = filename
        self._open_file()
        self._process = lambda x: zstd.decompress(x) if x else x

    def _open_file(self):
        self.file = open(self.filename, 'rb')
        self._records = mmap.mmap(
            self.file.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = self._records.size()

        (index_start,) = struct.unpack('<Q', self.mm.read(8))

        self.index_start = index_start
        self.num_records = (file_size - index_start) // 8

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        if not 0 <= idx < self._num_records:
            raise IndexError('BagReader index out of range')

        end = idx * 8 + self.index_start

        rec_range = struct.unpack('<2q', self._records[end - 8: end + 8])

        return self._process(self._records[slice(*rec_range)])

    def close(self):
        self.mm.close()
        self.file.close()
