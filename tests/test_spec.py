"""The recorded hash of the analysis spec must be the spec's (config/analysis.json, frozen_spec)."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Spec(unittest.TestCase):
    def test_last_amendment_hash_is_the_current_spec(self):
        record = json.loads((ROOT / 'config/analysis.json').read_text())['frozen_spec']
        current = hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest()
        self.assertEqual(record['amendments'][-1]['sha256_after'], current,
                         'docs/analysis-spec.md changed without a recorded amendment')

    def test_every_amendment_is_dated_and_says_what_existed(self):
        amendments = json.loads((ROOT / 'config/analysis.json').read_text())['frozen_spec']['amendments']
        dates = [a['date'] for a in amendments]
        self.assertEqual(dates, sorted(dates))
        self.assertTrue(all(a['results_existing'] and len(a['sha256_after']) == 64 for a in amendments))


if __name__ == '__main__':
    unittest.main()
