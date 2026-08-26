"""
Random access into the InsScene-15K mirror WITHOUT unpacking it (docs/DATASET.md §2.3).

The mirror stores two very different shapes, and this module hides the difference behind one
`iter_members`/`read` interface:

  processed_scannetpp_v2/   ONE 211 GiB zip cut into 53 fixed-size parts (`.zip.001`, ...).
                            It is a plain byte split, NOT a `zip -s` multi-disk archive, so the
                            central directory sits at the tail of the LAST part and every member
                            can be reached by seeking across the concatenation.
  processed_re10k/          the same shape: one 169 GiB zip in 43 parts, 1 221 783 members.
  processed_infinigen/      one ordinary ~60 MB zip per sub-scene, opened with `zipfile`.

**Why not just concatenate.** `cat *.zip.0* > all.zip` would materialise 211 GiB and we need
~0.3 % of the members (32 frames per scene). Reading the 500 MiB central directory once and then
seeking is two orders of magnitude cheaper, and costs no scratch inodes (docs/DATASET.md §5.1).

The zip64 handling is not optional here: both archives are past 4 GiB, so sizes and local-header
offsets live in the 0x0001 extra field and the 32-bit fields read 0xFFFFFFFF.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ZIP64_SENTINEL_32 = 0xFFFFFFFF
EOCD_SIG = b"PK\x05\x06"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CD_ENTRY_SIG = b"PK\x01\x02"


@dataclass(frozen=True)
class Member:
    """One archive entry, located in the concatenated byte stream of all parts."""

    name: str
    method: int          # 0 = stored, 8 = deflate; nothing else appears in this mirror
    compressed_size: int
    size: int
    header_offset: int   # offset of the local file header


class SplitZipReader:
    """A zip cut into fixed-size parts by `split -b`, read without joining the parts."""

    def __init__(self, directory: Path, stem: str):
        self.parts: List[Path] = sorted(
            directory.glob(f"{stem}.zip.*"),
            key=lambda p: int(p.suffix.lstrip(".")),
        )
        if not self.parts:
            raise FileNotFoundError(f"no {stem}.zip.NNN parts under {directory}")
        self.sizes = [p.stat().st_size for p in self.parts]
        self.total = sum(self.sizes)
        self._members: Dict[str, Member] = {}

    # ---- byte plumbing ---------------------------------------------------------------------

    def read_range(self, start: int, length: int) -> bytes:
        """Read `length` bytes at absolute offset `start`, crossing part boundaries."""
        if start < 0 or length < 0 or start + length > self.total:
            raise ValueError(f"range {start}+{length} outside the {self.total}-byte archive")
        out, part_start = bytearray(), 0
        for path, size in zip(self.parts, self.sizes):
            part_end = part_start + size
            if part_end > start and part_start < start + length:
                lo = max(start, part_start) - part_start
                hi = min(start + length, part_end) - part_start
                with path.open("rb") as fh:
                    fh.seek(lo)
                    chunk = fh.read(hi - lo)
                if len(chunk) != hi - lo:
                    raise IOError(f"short read in {path.name}: {len(chunk)} != {hi - lo}")
                out += chunk
            part_start = part_end
        return bytes(out)

    def _tail(self, n: int) -> bytes:
        return self.read_range(max(0, self.total - n), min(n, self.total))

    # ---- the central directory -------------------------------------------------------------

    def _locate_central_directory(self) -> Tuple[int, int]:
        tail = self._tail(1 << 20)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise ValueError("no end-of-central-directory record in the last part")
        cd_size, cd_offset = struct.unpack("<II", tail[i + 12:i + 20])
        j = tail.rfind(ZIP64_EOCD_SIG)
        if j >= 0:                                    # zip64 wins whenever it is present
            cd_size, cd_offset = struct.unpack("<QQ", tail[j + 40:j + 56])
        return cd_offset, cd_size

    @staticmethod
    def _zip64_overrides(extra: bytes, size: int, csize: int, offset: int):
        """Apply the 0x0001 extra field, which lists ONLY the fields that were sentinelled."""
        i = 0
        while i + 4 <= len(extra):
            tag, field_len = struct.unpack("<HH", extra[i:i + 4])
            if tag == 1:
                values, k = [], i + 4
                while k + 8 <= i + 4 + field_len:
                    values.append(struct.unpack("<Q", extra[k:k + 8])[0])
                    k += 8
                it = iter(values)
                if size == ZIP64_SENTINEL_32:
                    size = next(it, size)
                if csize == ZIP64_SENTINEL_32:
                    csize = next(it, csize)
                if offset == ZIP64_SENTINEL_32:
                    offset = next(it, offset)
                break
            i += 4 + field_len
        return size, csize, offset

    def members(self) -> Dict[str, Member]:
        """Parse (once) and return every entry, keyed by name. ~0.5 GiB read for ScanNet++."""
        if self._members:
            return self._members
        cd_offset, cd_size = self._locate_central_directory()
        cd = self.read_range(cd_offset, cd_size)
        out, i = {}, 0
        while i + 46 <= len(cd) and cd[i:i + 4] == CD_ENTRY_SIG:
            method, = struct.unpack("<H", cd[i + 10:i + 12])
            csize, size = struct.unpack("<II", cd[i + 20:i + 28])
            name_len, extra_len, comment_len = struct.unpack("<HHH", cd[i + 28:i + 34])
            offset, = struct.unpack("<I", cd[i + 42:i + 46])
            name = cd[i + 46:i + 46 + name_len].decode("utf-8", "replace")
            extra = cd[i + 46 + name_len:i + 46 + name_len + extra_len]
            size, csize, offset = self._zip64_overrides(extra, size, csize, offset)
            out[name] = Member(name, method, csize, size, offset)
            i += 46 + name_len + extra_len + comment_len
        if not out:
            raise ValueError("central directory parsed to zero entries")
        self._members = out
        return out

    # ---- member payloads -------------------------------------------------------------------

    def read(self, name: str) -> bytes:
        """Inflate one member. The local header's name/extra lengths may differ from the CD's."""
        member = self.members()[name]
        head = self.read_range(member.header_offset, 30)
        if head[:4] != b"PK\x03\x04":
            raise ValueError(f"{name}: no local file header at {member.header_offset}")
        name_len, extra_len = struct.unpack("<HH", head[26:30])
        raw = self.read_range(member.header_offset + 30 + name_len + extra_len,
                              member.compressed_size)
        if member.method == 0:
            data = raw
        elif member.method == 8:
            data = zlib.decompress(raw, -zlib.MAX_WBITS)
        else:
            raise ValueError(f"{name}: unsupported compression method {member.method}")
        if member.size and len(data) != member.size:
            raise IOError(f"{name}: inflated {len(data)} bytes, expected {member.size}")
        return data

    def iter_names(self, prefix: str = "", suffix: str = "") -> Iterator[str]:
        for name in sorted(self.members()):
            if name.startswith(prefix) and name.endswith(suffix):
                yield name


def scene_ids(reader: SplitZipReader, root: str, id_len: int = 10) -> List[str]:
    """The scene directories directly under `root/` (ScanNet++ ids are 10 hex characters)."""
    out = set()
    for name in reader.members():
        parts = name.split("/")
        if len(parts) > 2 and parts[0] == root and len(parts[1]) == id_len:
            if all(c in "0123456789abcdef" for c in parts[1]):
                out.add(parts[1])
    return sorted(out)
