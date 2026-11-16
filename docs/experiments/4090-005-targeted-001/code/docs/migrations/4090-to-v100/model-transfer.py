"""Transfer only the second-cycle model, root adapter files and organizer data."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import shlex
import tarfile

EXPECTED_WEIGHT_SHA256 = 'ef0bc5ae0646f6a4bda4cee69fe64b292d710e5b54119ac5ae9868720af584f7'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def find_model(explicit=None):
    roots = [Path(explicit)] if explicit else [Path.home() / name / 'out/merged-4090-continue-002'
                                              for name in ('mws', 'mws-measurements-002')]
    for root in roots:
        weights = root / 'model.safetensors'
        if weights.is_file() and digest(weights) == EXPECTED_WEIGHT_SHA256:
            for name in ('config.json', 'tokenizer_config.json', 'case_contract.json'):
                if not (root / name).is_file():
                    raise ValueError(f'Model auxiliary file missing: {name}')
            return root
    raise ValueError('Expected second-cycle merged model not found. Specify --model with its actual directory.')


def pack(output, model_dir=None, data_dir=None, adapter_dir=None):
    output = Path(output).absolute()
    model = find_model(model_dir)
    data = Path(data_dir) if data_dir else next((p for p in [model.parent.parent / 'data/organizer',
                   Path.home() / 'mws/data/organizer'] if (p / 'val.jsonl').is_file()), None)
    if data is None:
        raise ValueError('Organizer data not found; specify --data')
    sources = {}
    for path in sorted(model.rglob('*')):
        if path.is_file() and not any(p in ('.git', '.cache') for p in path.relative_to(model).parts):
            sources['models/continue-002/' + path.relative_to(model).as_posix()] = path
    for name in ('val.jsonl', 'catalog.json', 'train.jsonl'):
        path = data / name
        if not path.is_file():
            raise ValueError(f'Missing organizer file: {path}')
        sources['data/organizer/' + name] = path
    adapter_candidates = [Path(adapter_dir)] if adapter_dir else [model.parent / '4090-continue-002',
        Path.home() / 'mws-measurements-002/out/4090-continue-002', Path.home() / 'mws/out/4090-continue-002']
    adapter = next((p for p in adapter_candidates if (p / 'adapter_config.json').is_file()
                    and (p / 'adapter_model.safetensors').is_file()), adapter_candidates[0])
    adapter_saved = (adapter / 'adapter_config.json').is_file() and (adapter / 'adapter_model.safetensors').is_file()
    if adapter_dir and not adapter_saved:
        raise ValueError('Explicit adapter path lacks adapter_config.json/adapter_model.safetensors')
    if adapter_saved:
        # No checkpoint trees, optimizer state or historical training logs.
        for path in sorted(adapter.iterdir()):
            if path.is_file():
                sources['adapters/continue-002/' + path.name] = path
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path = Path(str(output) + '.manifest.json')
    partial = Path(str(output) + '.partial')
    if output.exists() or manifest_path.exists() or partial.exists():
        raise FileExistsError('Transfer files exist; choose a new output name')
    estimated = sum(p.stat().st_size for p in sources.values())
    if shutil.disk_usage(output.parent).free < estimated + 256 * 2**20:
        raise RuntimeError('Insufficient free disk space on source for this small targeted package')
    records = []
    for name, path in sources.items():
        print('Preparing:', name, flush=True)
        info = path.stat()
        records.append({'name': name, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
                        'sha256': digest(path), 'source': str(path)})
    internal = {'scope': 'merged cycle-2 model + root adapter files + train/val/catalog only',
                'expected_model_sha256': EXPECTED_WEIGHT_SHA256,
                'model_source': str(model), 'adapter_source': str(adapter) if adapter_saved else None,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'files': records}
    with tarfile.open(partial, mode='x:gz', compresslevel=1, dereference=True) as archive:
        for record in records:
            archive.add(record['source'], arcname=record['name'], recursive=False)
            after = Path(record['source']).stat()
            if (after.st_size, after.st_mtime_ns) != (record['bytes'], record['mtime_ns']):
                raise RuntimeError(f'Source changed while packing: {record["source"]}')
        payload = json.dumps(internal, ensure_ascii=False, indent=2).encode('utf-8')
        item = tarfile.TarInfo('MODEL_TRANSFER_MANIFEST.json')
        item.size, item.mode = len(payload), 0o600
        archive.addfile(item, io.BytesIO(payload))
    partial.rename(output)
    summary = {**internal, 'archive': output.name, 'archive_bytes': output.stat().st_size,
               'archive_sha256': digest(output), 'uncompressed_bytes': estimated}
    save(manifest_path, summary)
    print(json.dumps({'archive': str(output), 'manifest': str(manifest_path),
                      'bytes': summary['archive_bytes'], 'files': len(records)}, indent=2), flush=True)


def install(archive_path, root):
    archive_path, root = Path(archive_path), Path(root).resolve()
    if (root / 'out/v100-assets.json').exists() or (root / 'runtime.env').exists():
        raise FileExistsError('Preserve existing model runtime settings')
    summary = json.loads(Path(str(archive_path) + '.manifest.json').read_text(encoding='utf-8-sig'))
    if archive_path.stat().st_size != summary['archive_bytes'] or digest(archive_path) != summary['archive_sha256']:
        raise ValueError('Transferred model archive checksum mismatch')
    with tarfile.open(archive_path, mode='r:gz') as archive:
        internal = json.load(archive.extractfile('MODEL_TRANSFER_MANIFEST.json'))
        allowed = {f['name'] for f in internal['files']} | {'MODEL_TRANSFER_MANIFEST.json'}
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members) or {m.name for m in members} != allowed:
            raise ValueError('Unexpected archive entries')
        for item in members:
            name = PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts or not item.isfile():
                raise ValueError('Only regular relative files are allowed')
            target = root.joinpath(*name.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError('Path would escape the team directory')
            if target.exists():
                raise FileExistsError(f'Preserve existing target: {target}')
        if shutil.disk_usage(root).free < summary['uncompressed_bytes'] + 256 * 2**20:
            raise RuntimeError('Insufficient free disk space to unpack model')
        archive.extractall(root, filter='data')
    for record in internal['files']:
        if digest(root / record['name']) != record['sha256']:
            raise ValueError(f'Unpacked checksum mismatch: {record["name"]}')
    settings = {'model': str(root / 'models/continue-002'), 'val': str(root / 'data/organizer/val.jsonl'),
                'catalog': str(root / 'data/organizer/catalog.json'), 'train': str(root / 'data/organizer/train.jsonl'),
                'adapter': str(root / 'adapters/continue-002') if internal['adapter_source'] else None,
                'model_sha256': internal['expected_model_sha256'],
                'restored_hf_cache': str(root / '.cache/huggingface')}
    (root / 'out').mkdir(exist_ok=True)
    save(root / 'out/v100-assets.json', settings)
    runtime = {'MODEL_DIR': settings['model'], 'MERGED_DIR': settings['model'],
               'CATALOG_PATH': settings['catalog'], 'JSON_CONSTRAINED': '1',
               'JSON_GRAMMAR_BACKEND': 'torch_native', 'JSON_GRAMMAR_THREADS': '4',
               'TEMPORAL_NORMALIZATION': 'task', 'TEMPORAL_SERIES_HISTORY': '1',
               'MAX_INPUT_TOKENS': '12288', 'MAX_SEQUENCE_TOKENS': '16384',
               'MAX_NEW_TOKENS': '4096', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    with (root / 'runtime.env').open('x', encoding='utf-8') as stream:
        for key, value in runtime.items():
            stream.write(f'export {key}={shlex.quote(value)}\n')
    print('MODEL AND DATA INSTALLED AND VERIFIED:', root, flush=True)


def main():
    if os.name != 'nt':
        os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    source = commands.add_parser('pack')
    source.add_argument('--output', required=True)
    source.add_argument('--model')
    source.add_argument('--data')
    source.add_argument('--adapter')
    target = commands.add_parser('install')
    target.add_argument('--archive', required=True)
    target.add_argument('--root', required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.output, args.model, args.data, args.adapter)
    else:
        install(args.archive, args.root)


if __name__ == '__main__':
    main()
