import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from catalog_normalization import CatalogResolver, normalize_catalog_facts


class CatalogTest(unittest.TestCase):
    def test_exact_spelling_keeps_important_plus_sign(self):
        resolver = CatalogResolver({'tariffs': ['Анлим+', 'Анлим']})
        self.assertEqual(resolver.resolve(' «АНЛИМ+» ', 'tariffs').canonical, 'Анлим+')

    def test_ambiguous_alias_and_fuzzy_names_are_not_merged(self):
        resolver = CatalogResolver({'services': ['домашний интернет', 'мобильный интернет']},
                                   {'services': {'инет': ['домашний интернет', 'мобильный интернет']}})
        self.assertIsNone(resolver.resolve('инет', 'services').canonical)
        self.assertEqual(resolver.resolve('инет', 'services').reason, 'ambiguous')
        self.assertIsNone(resolver.resolve('интернет', 'services').canonical)

    def test_alias_must_reference_real_catalog_value(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            CatalogResolver({'devices': ['роутер']}, {'devices': {'модем': 'неизвестно'}})

    def test_type_and_ids_and_action_result_are_preserved(self):
        resolver = CatalogResolver({'devices': ['Роутер'], 'services': [], 'tariffs': ['Плюс']})
        fact = {'object': {'name': ' роутер ', 'type': 'device', 'id': 'own:dev'},
                'attribute': 'action', 'value': 'перезагрузка роутера — не помогло',
                'polarity': 1, 'source': {'turn_idx': 4}, 'valid_from': None, 'valid_to': None}
        saved = copy.deepcopy(fact)
        result, decisions = normalize_catalog_facts([fact], resolver)
        self.assertEqual(result[0]['object']['name'], 'Роутер')
        self.assertEqual(result[0]['object']['id'], 'own:dev')
        self.assertEqual(result[0]['value'], fact['value'])
        self.assertEqual(fact, saved)
        self.assertEqual(len(decisions), 1)

    def test_ticket_named_like_a_device_is_not_canonicalized(self):
        resolver = CatalogResolver({'devices': ['Роутер']})
        fact = {'object': {'name': 'роутер', 'type': 'ticket'}, 'attribute': 'ticket_status', 'value': 'принята'}
        self.assertEqual(normalize_catalog_facts([fact], resolver)[0], [fact])


if __name__ == '__main__':
    unittest.main()
