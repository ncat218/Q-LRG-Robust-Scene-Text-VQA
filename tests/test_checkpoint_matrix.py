import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from qlrg.common import object_hash, read_json, sha256, write_json
from qlrg.s1_model import VisualResidual
from scripts.checkpoint_matrix import report, validate_state
from scripts.checkpoint_matrix_plan import comparison_pairs, module_plan, resolve_training, specifications


class CheckpointMatrixTests(unittest.TestCase):
    def test_aliases_and_comparisons_keep_condition_budget_and_reader_intent(self):
        profiles = {
            'old11305': {'full': 'old', 'crop': 'old', 'readers': ['A0', 'A2']},
            'new_A0': {'full': 'f', 'crop': 'c', 'readers': ['A0', 'A2']},
            'new_A2_restoration': {'full': None, 'crop': 'rc', 'readers': ['A2']},
            'new_A2_ce_only': {'full': None, 'crop': 'cc', 'readers': ['A2']}}
        specs, table = specifications(profiles)
        self.assertEqual((len(specs), len(table)), (120, 192))
        names = {s['arm'] for s in specs}
        for left, right in comparison_pairs(table):
            self.assertIn(left, names); self.assertIn(right, names)
            self.assertEqual(left.split('__')[1:3], right.split('__')[1:3])
        # A selected zero full module is literally the same no-restoration arm.
        rows = [r for r in table if r['profile'] == 'new_A2_restoration' and r['layout'] == 'full']
        for row in rows:
            matching = [r for r in rows if r['reader'] == row['reader'] and r['condition'] == row['condition']
                        and r['budget'] == row['budget']]
            self.assertEqual(len({r['arm'] for r in matching}), 1)

    def test_no_roi_fallback_uses_requested_crop_module_but_never_fusion_crop_on_full(self):
        self.assertEqual(module_plan('crop', ('crop_trained',), False), (('full',), ('crop_trained',)))
        self.assertEqual(module_plan('fusion', (None, 'crop_trained'), False), (('full',), (None,)))
        self.assertEqual(module_plan('fusion', ('full_trained', 'crop_trained'), True),
                         (('full', 'crop'), ('full_trained', 'crop_trained')))

    def fixture(self, root, best=12, refit=True):
        directory = root / 'full'
        binding = {'base_config': {'hidden': 3, 'spatial_blocks': 1}}
        write_json(directory / 'run.json', binding)
        model = VisualResidual(4, 3, 1)
        selected_path = directory / 'selection/best.pt'
        selected_path.parent.mkdir(parents=True)
        torch.save({'binding': object_hash(binding), 'examples': best, 'adapter': model.state_dict()}, selected_path)
        selected = {'binding': object_hash(binding), 'checkpoint': 'best.pt',
                    'checkpoint_sha256': sha256(selected_path), 'selection': {'best_examples': best}}
        write_json(directory / 'selection/COMPLETE.json', selected)
        summary = {'selection': selected, 'refit': {'skipped': 'No winning trained checkpoint'}}
        if refit and best:
            refit_binding = {**binding, 'selected_examples': best,
                             'selected_checkpoint_sha256': selected['checkpoint_sha256'], 'refit_visits': 16,
                             'refit_rule': 'same selected fractional epochs over all official_train'}
            path = directory / 'refit/latest.pt'; path.parent.mkdir()
            torch.save({'binding': object_hash(refit_binding), 'arm': 'S1b', 'examples': 16,
                        'adapter': model.state_dict()}, path)
            summary['refit'] = {'binding': object_hash(refit_binding), 'checkpoint': 'latest.pt',
                                'checkpoint_sha256': sha256(path), 'examples': 16, 'epochs_completed': 1.}
            write_json(directory / 'refit/COMPLETE.json', summary['refit'])
        write_json(directory / 'summary.json', summary)
        write_json(directory / 'COMPLETE.json', {'binding': object_hash(binding),
                                                'summary_sha256': sha256(directory / 'summary.json')})
        write_json(root / 'COMPLETE.json', {'stage': 's1_convergence', 'views': ['full', 'crop'],
                                          'summaries': {'full': sha256(directory / 'summary.json')}})
        return selected_path

    def test_refit_required_and_tensor_state_binding_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            record = resolve_training(root, 'full')
            self.assertFalse(record['baseline_kept'])
            self.assertIn('output.weight', validate_state(record))
            changed = copy.deepcopy(record); changed['checkpoint_binding'] = 'other'
            with self.assertRaisesRegex(ValueError, 'does not match'):
                validate_state(changed)
            Path(record['source']).write_bytes(b'truncated checkpoint')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                resolve_training(root, 'full')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root, refit=False)
            with self.assertRaisesRegex(ValueError, 'requires completed'):
                resolve_training(root, 'full')

    def test_zero_baseline_is_not_mislabeled_as_a_trained_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.fixture(root, best=0)
            record = resolve_training(root, 'full')
            self.assertTrue(record['baseline_kept'])
            self.assertIsNone(validate_state(record))
            state = torch.load(record['selection_source'], weights_only=True)
            state['adapter']['output.bias'][0] = 1
            torch.save(state, record['selection_source'])
            with self.assertRaisesRegex(ValueError, 'nonzero'):
                validate_state(record)

    def test_report_rejects_missing_questions_and_wrong_fair_tokens(self):
        profiles = {'old11305': {'full': 'old', 'crop': 'old', 'readers': ['A0']}}
        specs, table = specifications(profiles)
        spec = specs[0]
        manifest = {'images': [{'image_id': 'im', 'questions': [{'question_id': 1}, {'question_id': 2}]}]}
        row = {**spec, 'image_id': 'im', 'question_id': 1}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'same ordered'):
                report(Path(tmp), [row], manifest, [spec], [], {}, [])
            rows = [{**row, 'question_id': q, 'visual_tokens': 575} for q in (1, 2)]
            with self.assertRaisesRegex(ValueError, 'exactly 576'):
                report(Path(tmp), rows, manifest, [spec], [], {}, [])


if __name__ == '__main__':
    unittest.main()
