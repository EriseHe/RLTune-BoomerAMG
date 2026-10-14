"""Resolve only the explicitly selected official paper datasets."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'experiments/paper_final/reproduction'


def registry():
    result = json.loads((EVIDENCE / 'manifest.json').read_text())
    if result['format'] != 'official-paper-data-v1' or set(result['modules']) != {'module04', 'module05', 'module06'}:
        raise ValueError('Unexpected official data registry')
    return result


def _local(relative):
    path = ROOT / relative
    resolved = path.resolve()
    if not resolved.is_relative_to((ROOT / 'data/official').resolve()):
        raise ValueError(f'Official data must not resolve into the archive: {relative}')
    if not path.exists():
        raise FileNotFoundError(f'Official local data is unavailable: {path}')
    return path


def module_root(module):
    return _local(registry()['modules'][module]['local_root'])


def runtime_root():
    return _local(registry()['runtime_root'])


def resolve_recorded_path(value):
    """Translate preserved absolute provenance paths without changing records.

    Mappings are exact directory prefixes from the accepted data registry. There
    is deliberately no discovery of similarly named runs or archive fallback.
    """
    path = Path(value)
    mapping = json.loads((EVIDENCE / 'source_paths.json').read_text())
    if path.is_relative_to(ROOT / 'data/official'):
        return _local(str(path.relative_to(ROOT)))
    for source, destination in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        source_path = Path(source)
        if path == source_path or path.is_relative_to(source_path):
            return _local(str(Path(destination) / path.relative_to(source_path)))
    raise ValueError(f'Path is not part of an official run: {value}')
