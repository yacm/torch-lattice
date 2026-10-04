"""Regresiones de endurecimiento para archivos LIME y artefactos publicados."""
import io
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tlattice import gauge, lime

ROOT = Path(__file__).resolve().parents[1]


def _small_lime(directory):
    path = Path(directory) / "field.lime"
    U = gauge.unit_gauge((1, 1, 1, 1), dtype=torch.complex128)
    lime.write_lime(U, path, (1, 1, 1, 1))
    return path


def _encode_records(records):
    stream = io.BytesIO()
    for typ, payload in records:
        lime._emit_record(stream, True, True, typ, payload)
    return stream.getvalue()


def _read_in_child(path, queue):
    try:
        lime.read_lime(path)
    except ValueError as error:
        queue.put((type(error).__name__, str(error)))
    else:
        queue.put(("accepted", ""))


def test_write_lime_does_not_follow_symlink():
    """Una salida controlada por otro usuario no debe sobrescribir el destino del symlink."""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        victim = directory / "victim.txt"
        victim.write_text("NO TOCAR", encoding="utf-8")
        output = directory / "output.lime"
        output.symlink_to(victim)
        U = gauge.unit_gauge((1, 1, 1, 1), dtype=torch.complex128)

        lime.write_lime(U, output, (1, 1, 1, 1))

        assert victim.read_text(encoding="utf-8") == "NO TOCAR"
        assert output.is_file() and not output.is_symlink()
        read_back, dims = lime.read_lime(output)
        assert dims == [1, 1, 1, 1]
        assert len(read_back) == 4


def test_iter_records_rejects_truncated_payload():
    header = lime._record_header(True, True, "truncated", 16)
    try:
        list(lime._iter_records(header + b"abc"))
    except ValueError as error:
        assert "truncado" in str(error).lower()
    else:
        raise AssertionError("un payload truncado fue aceptado")


def test_read_lime_enforces_explicit_size_limit():
    with tempfile.TemporaryDirectory() as directory:
        path = _small_lime(directory)
        try:
            lime.read_lime(path, max_bytes=32)
        except ValueError as error:
            assert "tamaño" in str(error).lower()
        else:
            raise AssertionError("read_lime ignoró max_bytes")


def test_read_lime_size_limit_uses_open_descriptor_and_bounded_read():
    """Un stat previo falso no debe permitir leer más que max_bytes."""
    with tempfile.TemporaryDirectory() as directory:
        path = _small_lime(directory)
        with mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=1)):
            try:
                lime.read_lime(path, max_bytes=32)
            except ValueError as error:
                assert "tamaño" in str(error).lower()
            else:
                raise AssertionError("el límite fue evadido entre stat y open")


def test_read_lime_rejects_non_regular_file():
    device = Path("/dev/null")
    if not device.exists():
        return
    try:
        lime.read_lime(device)
    except ValueError as error:
        assert "regular" in str(error).lower()
    else:
        raise AssertionError("read_lime aceptó un dispositivo")


def test_read_lime_rejects_fifo_without_blocking():
    if not hasattr(os, "mkfifo"):
        return
    with tempfile.TemporaryDirectory() as directory:
        fifo = Path(directory) / "field.fifo"
        os.mkfifo(fifo)
        queue = mp.Queue()
        process = mp.Process(target=_read_in_child, args=(fifo, queue))
        process.start()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError("read_lime quedó bloqueado al abrir un FIFO")
        kind, message = queue.get(timeout=1)
        assert kind == "ValueError" and "regular" in message.lower()


def test_read_lime_requires_checksum_when_verification_enabled():
    with tempfile.TemporaryDirectory() as directory:
        path = _small_lime(directory)
        records = [
            item for item in lime._iter_records(path.read_bytes())
            if item[0] != lime.REC_CHECKSUM
        ]
        path.write_bytes(_encode_records(records))
        try:
            lime.read_lime(path)
        except ValueError as error:
            assert "checksum" in str(error).lower()
        else:
            raise AssertionError("LIME sin checksum fue aceptado")


def test_read_lime_rejects_trailing_garbage():
    with tempfile.TemporaryDirectory() as directory:
        path = _small_lime(directory)
        path.write_bytes(path.read_bytes() + b"NOT-LIME-TRAILER")
        try:
            lime.read_lime(path)
        except ValueError as error:
            assert "cabecera" in str(error).lower() or "magic" in str(error).lower()
        else:
            raise AssertionError("LIME con basura final fue aceptado")


def test_read_lime_rejects_duplicate_records():
    with tempfile.TemporaryDirectory() as directory:
        path = _small_lime(directory)
        records = list(lime._iter_records(path.read_bytes()))
        checksum = next(item for item in records if item[0] == lime.REC_CHECKSUM)
        path.write_bytes(_encode_records([*records, checksum]))
        try:
            lime.read_lime(path)
        except ValueError as error:
            assert "duplicado" in str(error).lower()
        else:
            raise AssertionError("LIME con checksum duplicado fue aceptado")


def test_iter_records_rejects_invalid_message_boundaries():
    header = bytearray(lime._record_header(True, True, "record", 0))
    header[6] &= ~lime.MB_FLAG
    try:
        list(lime._iter_records(bytes(header)))
    except ValueError as error:
        assert "mensaje" in str(error).lower()
    else:
        raise AssertionError("registro inicial sin MB fue aceptado")


def test_read_lime_requires_all_canonical_records():
    with tempfile.TemporaryDirectory() as directory:
        path = _small_lime(directory)
        records = [
            item for item in lime._iter_records(path.read_bytes())
            if item[0] != lime.REC_FILE_XML
        ]
        path.write_bytes(_encode_records(records))
        try:
            lime.read_lime(path)
        except ValueError as error:
            assert lime.REC_FILE_XML in str(error)
        else:
            raise AssertionError("LIME sin metadatos canónicos fue aceptado")


def test_read_lime_rejects_checksum_xml_with_extra_structure():
    with tempfile.TemporaryDirectory() as directory:
        path = _small_lime(directory)
        records = []
        for typ, payload in lime._iter_records(path.read_bytes()):
            if typ == lime.REC_CHECKSUM:
                payload = payload.replace(
                    b"</scidacChecksum>", b"<unexpected/></scidacChecksum>"
                )
            records.append((typ, payload))
        path.write_bytes(_encode_records(records))
        try:
            lime.read_lime(path)
        except ValueError as error:
            assert "checksum" in str(error).lower()
        else:
            raise AssertionError("XML checksum con estructura extra fue aceptado")


def test_write_lime_rejects_field_dimension_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "field.lime"
        U = gauge.unit_gauge((1, 1, 1, 2), dtype=torch.complex128)
        try:
            lime.write_lime(U, output, (1, 1, 1, 1))
        except ValueError as error:
            assert "dimensiones" in str(error).lower()
        else:
            raise AssertionError("write_lime escribió campo/dimensiones incompatibles")
        assert not output.exists()


def test_iter_records_rejects_version_and_flags_invalidos():
    header = bytearray(lime._record_header(True, True, "record", 0))
    for index, value in ((5, 2), (6, 1)):
        corrupted = bytearray(header)
        corrupted[index] = value
        try:
            list(lime._iter_records(bytes(corrupted)))
        except ValueError:
            pass
        else:
            raise AssertionError("cabecera LIME inválida fue aceptada")


def test_write_lime_cleans_temporary_after_failure():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        output = directory / "field.lime"
        U = gauge.unit_gauge((1, 1, 1, 1), dtype=torch.complex128)
        original = lime._emit_record
        calls = 0

        def fail_on_second_record(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("fallo inyectado")
            return original(*args, **kwargs)

        with mock.patch.object(lime, "_emit_record", fail_on_second_record):
            try:
                lime.write_lime(U, output, (1, 1, 1, 1))
            except OSError:
                pass
            else:
                raise AssertionError("no se propagó el fallo de escritura")
        assert not output.exists()
        assert list(directory.iterdir()) == []


def test_repository_has_no_machine_specific_absolute_paths():
    checked = [ROOT / "README.md", ROOT / "PLAN.md", ROOT / "benchmarks/canonical/manifest.json"]
    checked.extend((ROOT / "scripts").glob("*.py"))
    checked.extend(path for path in (ROOT / "tests").glob("*.py") if path.name != Path(__file__).name)
    failures = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        if "/root/proyectos/" in text or '"/tmp/' in text or "'/tmp/" in text:
            failures.append(str(path.relative_to(ROOT)))
    assert not failures, f"rutas locales absolutas publicadas: {failures}"


def test_gitignore_blocks_common_secret_files():
    patterns = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    for required in (".env", ".env.*", "*.pem", "*.key", "credentials.json"):
        assert required in patterns, f"falta patrón preventivo {required}"


def test_dependencies_are_pinned():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package in ("defusedxml", "jax", "torch", "numpy", "matplotlib"):
        assert f'"{package}==' in pyproject, f"{package} no está fijado exactamente"
    assert (ROOT / "uv.lock").is_file(), "falta uv.lock"


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print("=== ENDURECIMIENTO DE SEGURIDAD OK ===")


if __name__ == "__main__":
    main()
