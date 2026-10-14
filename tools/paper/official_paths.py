"""Select the official data registry; never discover runs in an archive."""
from pathlib import Path
import json


def registry(code):
    root = Path(code) / 'experiments/paper_final/reproduction'
    data = json.loads((root / 'manifest.json').read_text())
    assert data['format'] == 'official-paper-data-v1'
    assert set(data['modules']) == {'module04', 'module05', 'module06'}
    return data


def module_root(code, module):
    code = Path(code).resolve()
    path = code / registry(code)['modules'][module]['local_root']
    assert path.is_dir() and path.resolve().is_relative_to(code / 'data/official')
    return path


def runtime_root(code):
    code = Path(code).resolve()
    path = code / registry(code)['runtime_root']
    assert path.is_dir() and path.resolve().is_relative_to(code / 'data/official')
    return path


def recorded_path(code, value):
    code, path = Path(code).resolve(), Path(value)
    mapping = json.loads((code / 'experiments/paper_final/reproduction/source_paths.json').read_text())
    for source, destination in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        old = Path(source)
        if path == old or path.is_relative_to(old):
            result = code / destination / path.relative_to(old)
            assert result.exists() and result.resolve().is_relative_to(code / 'data/official')
            return result
    if path.is_relative_to(code / 'data/official') and path.exists():
        return path
    raise ValueError(f'Not an official data source: {value}')
