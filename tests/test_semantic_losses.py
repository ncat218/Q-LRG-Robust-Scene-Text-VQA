import unittest
from unittest.mock import patch
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from scripts.semantic_losses import causal_answer_logits, clean_answer_kd, patch_infonce, semantic_objective, valid_resume_offset
from qlrg.s1_model import VisualResidual


class SemanticLossTests(unittest.TestCase):
    def test_objective_changes_only_auxiliary_terms_and_gradients_reach_restorer(self):
        torch.manual_seed(42)
        base = torch.randn(4, 4)
        clean = torch.tensor([[0., 3., 0., 0.]]).expand(4, 4).clone()
        cache = SimpleNamespace(features={'LR4': base, 'CLEAN': clean}, grid=(2, 2),
                                mask=torch.tensor([True, True, False, False]),
                                prompt=lambda c, q: {'condition': c, 'question': q})
        qwen = SimpleNamespace(dtype=torch.float32,
                    processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **k: [1], eos_token_id=2)))
        adapter = VisualResidual(4, 3, 1)
        config = {'max_answer_tokens': 32, 'feature_weight': 1., 'residual_weight': .01,
                  'clean_residual_weight': .1, 'kd_temperature': 2., 'kd_weight': .5,
                  'contrast_anchors': 128, 'contrast_temperature': .1, 'contrast_weight': .1,
                  'equivalent_cosine': .98}
        calls = []
        def logits(model, prompt, feature, targets):
            self.assertEqual(prompt['question'], 'Who?')
            self.assertEqual(targets, [1, 2])
            calls.append((prompt['condition'], torch.is_grad_enabled()))
            if prompt['condition'] == 'CLEAN':
                return feature.new_tensor([[0., 4., 0.], [0., 0., 4.]])
            return feature.mean(0)[:3].expand(2, 3)
        with patch('scripts.semantic_losses.causal_answer_logits', side_effect=logits):
            results = {}
            for method in ('continued', 'answer_kd', 'answer_kd_patch'):
                loss, terms, info = semantic_objective(qwen, adapter, cache, {'question': 'Who?'}, 'LR4', 'x', config, method)
                results[method] = terms
                grads = torch.autograd.grad(loss, tuple(adapter.parameters()), allow_unused=True)
                self.assertGreater(sum(float(g.norm()) for g in grads if g is not None), 0)
                if method != 'continued':
                    self.assertEqual(info['kd_positions'], 2)
            for key in ('task', 'feature', 'regularization'):
                torch.testing.assert_close(results['continued'][key], results['answer_kd'][key])
            self.assertNotIn('answer_kd', results['continued'])
            self.assertIn('patch', results['answer_kd_patch'])
        self.assertEqual([c for c in calls if c[0] == 'CLEAN'], [('CLEAN', False), ('CLEAN', False)])

    def test_resume_handles_short_accumulation_group_at_epoch_boundary(self):
        for offset in (0, 4, 31128, 31129, 31133, 62257, 62258):
            self.assertTrue(valid_resume_offset(offset, 31129, 4), offset)
        for offset in (-1, 1, 31130, 62259):
            self.assertFalse(valid_resume_offset(offset, 31129, 4), offset)
    def test_kd_teacher_detached_and_wrong_teacher_positions_are_excluded(self):
        student = torch.tensor([[.2, .1, -.1], [.1, .3, .2]], requires_grad=True)
        teacher = torch.tensor([[4., 0., -1.], [4., 0., -1.]], requires_grad=True)
        loss, info = clean_answer_kd(student, teacher, [0, 2], 2.)
        loss.backward()
        self.assertEqual(info['kd_positions'], 1)
        self.assertFalse(info['kd_eos_eligible'])
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad[0].norm()), 0)
        self.assertEqual(float(student.grad[1].norm()), 0)
        zero, info = clean_answer_kd(student, teacher, [1, 2], 2.)
        self.assertEqual(float(zero.detach()), 0.)
        self.assertTrue(torch.isfinite(zero))

    def test_patch_alignment_beats_permutation_and_does_not_backprop_to_teacher(self):
        clean = torch.eye(8, requires_grad=True)
        aligned = (torch.eye(8) + .01).requires_grad_()
        bad = aligned.roll(1, 0)
        mask = torch.tensor([True, True, False, False, False, False, False, False])
        a, stats = patch_infonce(aligned, clean, mask, 4, .1, .98)
        b, _ = patch_infonce(bad, clean, mask, 4, .1, .98)
        self.assertLess(float(a.detach()), float(b.detach()))
        self.assertEqual(stats['contrast_anchors'], 2)
        self.assertTrue(stats['contrast_text_anchors'])
        a.backward()
        self.assertIsNone(clean.grad)
        self.assertGreater(float(aligned.grad.norm()), 0)

    def test_repeated_teacher_tokens_are_not_forced_apart_and_missing_ocr_keeps_image(self):
        feature = torch.ones(3, 4, requires_grad=True)
        loss, stats = patch_infonce(feature, feature.detach(), None, 128, .1, .98)
        self.assertEqual(stats['contrast_valid_negatives'], 0)
        self.assertFalse(stats['contrast_text_anchors'])
        self.assertEqual(float(loss.detach()), 0.)
        small, _ = patch_infonce(feature[:1], feature[:1].detach(), None, 128, .1, .98)
        self.assertTrue(torch.isfinite(small))

    def test_causal_answer_positions_keep_query_and_eos_and_feature_gradient(self):
        seen = {}
        feature = torch.randn(4, 3, requires_grad=True)
        @contextmanager
        def replacement(batch, value):
            seen['feature'] = value
            yield
        def forward(**batch):
            seen.update(batch)
            return SimpleNamespace(logits=seen['feature'].mean(0).expand(1, len(batch['logits_to_keep']), 3))
        qwen = SimpleNamespace(model=forward, replacement=replacement)
        prompt = {'input_ids': torch.tensor([[10, 11, 12]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}
        logits = causal_answer_logits(qwen, prompt, feature, [1, 2])
        self.assertEqual(seen['input_ids'].tolist(), [[10, 11, 12, 1, 2]])
        self.assertEqual(seen['logits_to_keep'].tolist(), [2, 3])
        self.assertEqual(seen['attention_mask'].shape, (1, 5))
        logits.square().mean().backward()
        self.assertGreater(float(feature.grad.norm()), 0)


if __name__ == '__main__':
    unittest.main()
