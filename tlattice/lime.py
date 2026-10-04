"""Lector/escritor LIME (formato de configuraciones de Chroma/QIO, SciDAC/ILDG).

Bit-compatible con el stack USQCD. Formato verificado contra read_plaq.cc:
- cabecera LIME de 144 bytes (magic 0x456789ab, version, flags MB/ME, len, tipo)
- datos binarios: big-endian, doble precisión
- sitio lexicográfico (x0 más rápido, x3 más lento)
- matriz ColorMatrix en orden fila-mayor (i,j), cada elemento (re, im)
- checksum SciDAC: CRC32 (zlib) + rotación por rank global (rank%29 / rank%31)

U se devuelve/espera como lista de Nd tensores de forma (L0,L1,L2,L3,3,3),
donde el eje `mu` coincide con la dirección `mu` (roll en eje mu = shift en mu).
"""
import os
import re
import stat
import struct
import tempfile
import zlib
from math import prod
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from defusedxml import ElementTree

LIME_MAGIC = 0x456789AB
MB_FLAG = 0x80
ME_FLAG = 0x40
HEADER_SIZE = 144
DEFAULT_MAX_LIME_BYTES = 1024**3  # 1 GiB; ampliar explícitamente para retículas grandes

# records de la estructura QIO (mismo orden que escribe Chroma)
REC_PRIV_FILE_XML = "scidac-private-file-xml"
REC_FILE_XML = "scidac-file-xml"
REC_PRIV_REC_XML = "scidac-private-record-xml"
REC_REC_XML = "scidac-record-xml"
REC_ILDG_FORMAT = "ildg-format"
REC_BINARY = "ildg-binary-data"
REC_CHECKSUM = "scidac-checksum"


# --------------------------------------------------------------------------
# checksum
# --------------------------------------------------------------------------
def _rotl32(x, n):
    n &= 31
    if n == 0:
        return x & 0xFFFFFFFF
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def _crc32(b):
    return zlib.crc32(b) & 0xFFFFFFFF


def scidac_checksum(binary):
    """(suma, sumb) del checksum SciDAC sobre los datos binarios (big-endian)."""
    suma = sumb = 0
    n_site = len(binary) // 576  # 4 dirs x 18 doubles x 8 bytes
    for rank in range(n_site):
        w = _crc32(binary[rank * 576:(rank + 1) * 576])
        suma ^= _rotl32(w, rank % 29)
        sumb ^= _rotl32(w, rank % 31)
    return suma, sumb


# --------------------------------------------------------------------------
# bajo nivel LIME
# --------------------------------------------------------------------------
def _record_header(mb, me, type_str, data_len):
    flags = (MB_FLAG if mb else 0) | (ME_FLAG if me else 0)
    hdr = struct.pack(">I", LIME_MAGIC)          # magic
    hdr += struct.pack(">H", 1)                   # version
    hdr += bytes([flags, 0])                      # flags + reservado
    hdr += struct.pack(">Q", data_len)            # longitud de datos
    hdr += type_str.encode("ascii").ljust(128, b"\x00")
    if len(hdr) != HEADER_SIZE:
        raise RuntimeError(f"cabecera LIME interna de tamaño inválido: {len(hdr)}")
    return hdr


def _iter_records(data):
    """Genera (type_str, payload_bytes) recorriendo los registros LIME."""
    off = 0
    message_open = False
    while off < len(data):
        if len(data) - off < HEADER_SIZE:
            raise ValueError(
                f"cabecera LIME truncada: quedan {len(data) - off} bytes"
            )
        magic = struct.unpack(">I", data[off:off + 4])[0]
        if magic != LIME_MAGIC:
            raise ValueError(f"magic LIME inválido en offset {off}: {magic:#x}")
        version = struct.unpack(">H", data[off + 4:off + 6])[0]
        flags = data[off + 6]
        reserved = data[off + 7]
        if version != 1:
            raise ValueError(f"versión LIME no soportada: {version}")
        if flags & ~(MB_FLAG | ME_FLAG) or reserved != 0:
            raise ValueError(
                f"flags/reservado LIME inválidos: {flags:#x}/{reserved:#x}"
            )
        mb = bool(flags & MB_FLAG)
        me = bool(flags & ME_FLAG)
        if mb == message_open:
            state = "MB inesperado" if mb else "falta MB"
            raise ValueError(f"límites de mensaje LIME inválidos: {state}")
        message_open = not me
        dlen = struct.unpack(">Q", data[off + 8:off + 16])[0]
        try:
            typ = data[off + 16:off + 144].split(b"\x00")[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("tipo de registro LIME no ASCII") from error
        if not typ:
            raise ValueError("tipo de registro LIME vacío")
        payload_end = off + HEADER_SIZE + dlen
        if payload_end > len(data):
            raise ValueError(
                f"registro LIME truncado: {typ!r} declara {dlen} bytes, "
                f"quedan {len(data) - off - HEADER_SIZE}"
            )
        payload = data[off + HEADER_SIZE:payload_end]
        yield typ, payload
        padded_end = payload_end + ((8 - payload_end % 8) % 8)
        if padded_end > len(data):
            raise ValueError(f"padding LIME truncado después de {typ!r}")
        if any(data[payload_end:padded_end]):
            raise ValueError(f"padding LIME no nulo después de {typ!r}")
        off = padded_end
    if message_open:
        raise ValueError("mensaje LIME sin registro final ME")


def _emit_record(f, mb, me, type_str, payload):
    f.write(_record_header(mb, me, type_str, len(payload)))
    f.write(payload)
    pad = (8 - len(payload) % 8) % 8
    if pad:
        f.write(b"\x00" * pad)


# --------------------------------------------------------------------------
# conversión campo <-> binario
# --------------------------------------------------------------------------
def _binary_to_field(binary, dims):
    """bytes big-endian -> U (lista de Nd tensores complex128 (L0,L1,L2,L3,3,3))."""
    d = np.frombuffer(binary, dtype=">f8").reshape(-1, 4, 3, 3, 2)
    z = d[..., 0] + 1j * d[..., 1]                       # (Nsite, 4, 3, 3)
    L0, L1, L2, L3 = dims
    z = z.reshape(L3, L2, L1, L0, 4, 3, 3)               # x0 más rápido
    z = z.transpose(3, 2, 1, 0, 4, 5, 6)                 # -> (L0,L1,L2,L3,mu,i,j)
    return [torch.as_tensor(np.ascontiguousarray(z[..., mu, :, :]),
                            dtype=torch.complex128) for mu in range(4)]


def _field_to_binary(U):
    """U (lista de Nd tensores (L0,L1,L2,L3,3,3)) -> bytes big-endian."""
    flat = []
    for mu in range(len(U)):
        # (L0,L1,L2,L3,3,3) -> (L3,L2,L1,L0,3,3) -> plano lexicográfico x0 rápido
        t = U[mu].permute(3, 2, 1, 0, 4, 5).contiguous().reshape(-1, 3, 3)
        flat.append(t)
    stacked = torch.stack(flat, dim=1)                   # (Nsite, 4, 3, 3)
    arr = stacked.detach().cpu().numpy()
    f = arr.reshape(-1, 4, 9).view(np.float64).astype(">f8")
    return f.tobytes()


# --------------------------------------------------------------------------
# XML
# --------------------------------------------------------------------------
def _file_xml(dims):
    d = " ".join(str(x) for x in dims)
    return ('<?xml version="1.0" encoding="UTF-8"?><scidacFile>'
            '<version>1.1</version><spacetime>4</spacetime>'
            f'<dims>{d} </dims><volfmt>0</volfmt></scidacFile>\x00').encode()


def _file_xml_hb(dims):
    d = " ".join(str(x) for x in dims)
    return (f'<?xml version="1.0"?>\n\n\n<HB>\n  <ProgramInfo>\n'
            f'    <code_version>\n      <chroma>3.44.0</chroma>\n'
            f'      <qdp>1.46.0</qdp>\n      <basePrecision>64</basePrecision>\n'
            f'    </code_version>\n    <Setgeom>\n      <latt_size>{d}</latt_size>\n'
            f'      <total_volume>{np.prod(dims)}</total_volume>\n'
            f'    </Setgeom>\n  </ProgramInfo>\n</HB>\n\x00').encode()


def _record_xml_private(precision):
    return ('<?xml version="1.0" encoding="UTF-8"?><scidacRecord><version>1.1</version>'
            f'<recordtype>0</recordtype><datatype>QDP_D3_ColorMatrix</datatype>'
            f'<precision>{precision}</precision><colors>3</colors><spins>1</spins>'
            '<typesize>144</typesize><datacount>4</datacount></scidacRecord>\x00').encode()


def _record_xml(dims):
    d = " ".join(str(x) for x in dims)
    return (f'<?xml version="1.0"?>\n\n\n<ChromaHB>\n  <HBItr>\n    <nrow>{d}</nrow>'
            f'  </HBItr>\n</ChromaHB>\n\x00').encode()


def _ildg_format_xml(dims):
    L0, L1, L2, L3 = dims
    return ('<?xml version="1.0" encoding="UTF-8"?><ildgFormat '
            'xmlns="http://www.lqcd.org/ildg" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://www.lqcd.org/ildg/filefmt.xsd">'
            f'<version>1.0</version><field>su3gauge</field><precision>64</precision>'
            f'<lx>{L0}</lx><ly>{L1}</ly><lz>{L2}</lz><lt>{L3}</lt></ildgFormat>\x00').encode()


def _checksum_xml(suma, sumb):
    return ('<?xml version="1.0" encoding="UTF-8"?><scidacChecksum><version>1.0</version>'
            f'<suma>{suma:08x}</suma><sumb>{sumb:08x}</sumb></scidacChecksum>\x00').encode()


def _parse_xml(payload, label):
    """Parsea XML pequeño sin DTD/entidades y sin contenido tras la raíz."""
    try:
        text = payload.decode("utf-8").rstrip("\x00\t\r\n ")
    except UnicodeDecodeError as error:
        raise ValueError(f"XML {label} no es UTF-8") from error
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise ValueError(f"XML {label} contiene DTD o entidades no permitidas")
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise ValueError(f"XML {label} malformado") from error


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _validated_dims(dims):
    if (
        len(dims) != 4
        or any(not isinstance(value, Integral) or isinstance(value, bool) for value in dims)
        or any(value <= 0 for value in dims)
    ):
        raise ValueError(f"dimensiones LIME inválidas: {dims}")
    return [int(value) for value in dims]


def _dims_from_private_xml(payload):
    root = _parse_xml(payload, "de metadatos LIME")
    dims_nodes = [node for node in root.iter() if _local_name(node.tag) == "dims"]
    if len(dims_nodes) != 1 or dims_nodes[0].text is None:
        raise ValueError("metadatos LIME sin dimensiones únicas")
    try:
        dims = [int(value) for value in dims_nodes[0].text.split()]
    except ValueError as error:
        raise ValueError("dimensiones LIME no enteras") from error
    return _validated_dims(dims)


def _checksum_from_xml(payload):
    root = _parse_xml(payload, "de checksum LIME")
    children = list(root)
    expected_tags = ["version", "suma", "sumb"]
    if (
        _local_name(root.tag) != "scidacChecksum"
        or root.attrib
        or [_local_name(node.tag) for node in children] != expected_tags
        or any(node.attrib or list(node) for node in children)
        or (root.text or "").strip()
        or any((node.tail or "").strip() for node in children)
        or (children[0].text or "").strip() != "1.0"
    ):
        raise ValueError("registro de checksum LIME malformado")
    values = [(node.text or "").strip() for node in children[1:]]
    if any(re.fullmatch(r"[0-9a-fA-F]{8}", value) is None for value in values):
        raise ValueError("registro de checksum LIME malformado")
    return tuple(int(value, 16) for value in values)


# --------------------------------------------------------------------------
# API pública
# --------------------------------------------------------------------------
def read_lime(path, *, max_bytes=DEFAULT_MAX_LIME_BYTES, verify_checksum=True):
    """Lee un archivo LIME de gauge. Devuelve U (lista Nd de tensores complex128)."""
    path = Path(path)
    if max_bytes is not None and (
        not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0
    ):
        raise ValueError("max_bytes debe ser un entero no negativo o None")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"no se pudo abrir un archivo LIME regular: {path}") from error
    with os.fdopen(fd, "rb") as stream:
        descriptor = os.fstat(stream.fileno())
        if not stat.S_ISREG(descriptor.st_mode):
            raise ValueError("la entrada LIME debe ser un archivo regular")
        if max_bytes is not None and descriptor.st_size > max_bytes:
            raise ValueError(
                f"tamaño LIME {descriptor.st_size} excede el límite {max_bytes}"
            )
        data = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        if max_bytes is not None and len(data) > max_bytes:
            raise ValueError(f"tamaño LIME excede el límite {max_bytes} durante lectura")
    recs = {}
    for typ, payload in _iter_records(data):
        if typ in recs:
            raise ValueError(f"registro LIME duplicado: {typ!r}")
        recs[typ] = payload
    required = {
        REC_PRIV_FILE_XML,
        REC_FILE_XML,
        REC_PRIV_REC_XML,
        REC_REC_XML,
        REC_ILDG_FORMAT,
        REC_BINARY,
    }
    if verify_checksum:
        required.add(REC_CHECKSUM)
    missing = required - recs.keys()
    if missing:
        raise ValueError(f"registros LIME obligatorios ausentes: {sorted(missing)}")
    dims = _dims_from_private_xml(recs[REC_PRIV_FILE_XML])
    binary = recs[REC_BINARY]
    expected_bytes = prod(dims) * 576
    if len(binary) != expected_bytes:
        raise ValueError(
            f"payload gauge de {len(binary)} bytes; se esperaban {expected_bytes}"
        )
    if verify_checksum:
        expected_checksum = _checksum_from_xml(recs[REC_CHECKSUM])
        actual_checksum = scidac_checksum(binary)
        if actual_checksum != expected_checksum:
            raise ValueError(
                f"checksum LIME inválido: {actual_checksum!r} != {expected_checksum!r}"
            )
    return _binary_to_field(binary, dims), dims


def write_lime(U, path, dims, precision="D"):
    """Escribe LIME atómicamente, sin seguir symlinks en la ruta de salida."""
    dims = _validated_dims(dims)
    if precision != "D":
        raise ValueError("solo se admite precision='D' (complex128)")
    if len(U) != 4:
        raise ValueError(f"el campo gauge debe tener 4 direcciones, recibió {len(U)}")
    expected_shape = (*dims, 3, 3)
    for mu, link in enumerate(U):
        if not isinstance(link, torch.Tensor):
            raise TypeError(f"U[{mu}] no es un tensor")
        if tuple(link.shape) != expected_shape:
            raise ValueError(
                f"dimensiones de U[{mu}] {tuple(link.shape)} != {expected_shape}"
            )
        if link.dtype != torch.complex128:
            raise ValueError(f"U[{mu}] debe ser complex128 para precision='D'")
    binary = _field_to_binary(U)
    suma, sumb = scidac_checksum(binary)
    path = Path(path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as f:
            _emit_record(f, True,  False, REC_PRIV_FILE_XML, _file_xml(dims))
            _emit_record(f, False, True,  REC_FILE_XML,      _file_xml_hb(dims))
            _emit_record(f, True,  False, REC_PRIV_REC_XML,  _record_xml_private(precision))
            _emit_record(f, False, False, REC_REC_XML,       _record_xml(dims))
            _emit_record(f, False, False, REC_ILDG_FORMAT,   _ildg_format_xml(dims))
            _emit_record(f, False, False, REC_BINARY,        binary)
            _emit_record(f, False, True,  REC_CHECKSUM,      _checksum_xml(suma, sumb))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return suma, sumb
