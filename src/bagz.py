import struct
import mmap


class BagReader:
    def __init__(self, filename):
        self.filename = filename
        self._open_file()

    def _open_file(self):
        self.file = open(self.filename, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        self.mm.seek(-8, 2)
        (index_start,) = struct.unpack('<Q', self.mm.read(8))

        self.index_start = index_start
        self.num_records = (len(self.mm) - index_start) // 8

        self.offsets = []
        self.mm.seek(index_start)
        for _ in range(self.num_records):
            (offset,) = struct.unpack('<Q', self.mm.read(8))
            self.offsets.append(offset)

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        start = self.offsets[idx]
        end = self.offsets[idx + 1] if idx + \
            1 < self.num_records else self.index_start
        self.mm.seek(start)
        return self.mm.read(end - start)

    def close(self):
        self.mm.close()
        self.file.close()
