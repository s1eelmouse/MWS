"""Isolated versioned organizer scorers; official files remain byte-identical."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid

from config import SCORER_VERSION

FILES = {'1': 'official_case_metrics.py', '2.1': 'official_case_metrics_v21.py',
         '2.1-datefix': 'official_case_metrics_v21.py'}


class Scorer:
    def __init__(self, version=None, catalog_path=None):
        self.version = version or SCORER_VERSION
        if self.version not in FILES:
            raise ValueError('Unknown scorer version')
        self.path = Path(__file__).with_name(FILES[self.version])
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != self.path.with_suffix('.sha256').read_text().strip():
            raise ValueError('Organizer scorer bytes differ from the pinned checksum')
        name = '_isolated_case_scorer_' + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, self.path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.catalog_sha256 = None
        if self.catalog_path:
            self.catalog_sha256 = hashlib.sha256(self.catalog_path.read_bytes()).hexdigest()
        if self.version != '1':
            # Avoid CWD lookup and cross-experiment global catalogue/cache leakage.
            self.module._CATALOG = json.loads(self.catalog_path.read_text(encoding='utf-8')) if self.catalog_path else {}
            self.module._canon.cache_clear()
        if self.version == '2.1-datefix':
            tree = ast.parse(self.path.read_text(encoding='utf-8'))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == 'recover_dates')
            changed = 0
            for node in ast.walk(function):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'out'
                        and node.func.attr == 'update' and len(node.args) > 1):
                    node.args = [ast.List(elts=node.args, ctx=ast.Load())]
                    changed += 1
            if changed != 2:
                raise ValueError('Pinned datefix no longer matches exactly two intended branches')
            patch = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
            exec(compile(patch, str(self.path) + '+local-datefix1', 'exec'), self.module.__dict__)

    def evaluate(self, gold, predictions):
        return self.module.evaluate(gold, predictions)

    def match_facts(self, gold, predictions):
        return self.module.match_facts(gold, predictions)

    def temporal_eligible(self, fact, call):
        if self.version == '1':
            return True
        if fact['attribute'] not in self.module.TEMPORAL_ATTRS:
            return False
        from datetime import datetime
        anchor = datetime.fromisoformat(call['started_at']).date()
        recovered = set()
        for turn in call['turns']:
            recovered.update(self.module.recover_dates(turn['text'], anchor))
        gold_days = {v[:10] for v in (fact.get('valid_from'), fact.get('valid_to')) if v}
        return bool(gold_days & {d.isoformat() for d in recovered})

    def metadata(self):
        patched = self.version == '2.1-datefix'
        return {'scorer_version': self.version, 'official_metrics_version': '1' if self.version == '1' else '2.1',
                'scorer_sha256': self.sha256, 'official_source_modified': False,
                'local_patch': 'date-set-update-fix-1' if patched else None,
                'patch_adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if patched else None,
                'catalog_sha256': self.catalog_sha256}


def make_scorer(version=None, catalog_path=None):
    value = Scorer(version, catalog_path)
    return value
