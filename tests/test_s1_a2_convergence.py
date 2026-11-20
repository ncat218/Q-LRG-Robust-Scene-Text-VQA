import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from qlrg.s1_model import LanguageLoRA, LoRALinear, VisualResidual
from scripts.s1_a2_convergence import (assert_fixed_reader, baseline_once,
                                      method_config, module_digest, validate_study)
from scripts.s1_convergence import restoration_objective


class A2ConvergenceTests(unittest.TestCase):
    def fake_qwen(self):
        torch.manual_seed(42)
        attention=nn.Module()
        attention.q_proj=nn.Linear(4,4);attention.v_proj=nn.Linear(4,4)
        layer=nn.Module();layer.self_attn=attention
        language=nn.Module();language.layers=nn.ModuleList([layer])
        frozen=nn.Module();frozen.language_model=language;frozen.requires_grad_(False)
        lora=LanguageLoRA(language,2,4).requires_grad_(False)
        with torch.no_grad():
            for delta in lora.deltas:delta.up.weight.fill_(.1)
        tokenizer=SimpleNamespace(encode=lambda text,add_special_tokens:[1],eos_token_id=2)
        model=SimpleNamespace(model=frozen,dtype=torch.float32,processor=SimpleNamespace(tokenizer=tokenizer))
        return model,lora,attention

    def test_a2_stays_frozen_but_answer_gradient_reaches_restorer(self):
        model,lora,attention=self.fake_qwen();digest=module_digest(lora)
        adapter=VisualResidual(4,3,1)
        base=torch.randn(4,4)
        cache=SimpleNamespace(features={'LR4':base,'CLEAN':base+.5},grid=(2,2),prompt=lambda *args:{})
        config={'max_answer_tokens':8,'feature_weight':1.,'residual_weight':.01,'clean_residual_weight':.1}

        def teacher(qwen,prompt,feature,tokens):
            self.assertIsInstance(attention.q_proj,LoRALinear)
            return (attention.q_proj(feature)+attention.v_proj(feature)).square().mean()

        with lora.attached(),patch('scripts.s1_convergence.teacher_forced_loss',side_effect=teacher):
            loss,*_=restoration_objective(model,adapter,cache,{'question':'Who?'},'LR4','wallace',config)
            loss.backward()
            self.assertGreater(float(adapter.output.weight.grad.norm()),0.)
            assert_fixed_reader(model,lora,digest)
        self.assertIsInstance(attention.q_proj,nn.Linear)
        self.assertEqual(digest,module_digest(lora))
        self.assertTrue(all(p.grad is None for p in model.model.parameters()))
        self.assertTrue(all(p.grad is None for p in lora.parameters()))

    def test_ce_only_gradient_does_not_depend_on_clean_teacher_features(self):
        model,lora,attention=self.fake_qwen()
        adapter=VisualResidual(4,3,1)
        base=torch.randn(4,4)
        cache=SimpleNamespace(features={'LR4':base,'CLEAN':base+.5},grid=(2,2),prompt=lambda *args:{})
        original={'max_answer_tokens':8,'feature_weight':1.,'residual_weight':.01,'clean_residual_weight':.1}

        def teacher(qwen,prompt,feature,tokens):
            return (attention.q_proj(feature)+attention.v_proj(feature)).square().mean()

        gradients={}
        with lora.attached(),patch('scripts.s1_convergence.teacher_forced_loss',side_effect=teacher):
            for method in ('ce_only','restoration'):
                values=[]
                for shift in (.5,2.):
                    cache.features['CLEAN']=base+shift
                    loss,*_=restoration_objective(model,adapter,cache,{'question':'Read?'},'LR4','71',method_config(original,method))
                    values.append(torch.autograd.grad(loss,adapter.output.bias)[0])
                gradients[method]=values
        torch.testing.assert_close(*gradients['ce_only'],atol=0,rtol=0)
        self.assertFalse(torch.equal(*gradients['restoration']))
        self.assertEqual(original['feature_weight'],1.)
        self.assertEqual(method_config(original,'ce_only')['clean_residual_weight'],.1)

    def test_reader_guard_detects_trainable_or_changed_a2(self):
        model,lora,_=self.fake_qwen();digest=module_digest(lora)
        next(lora.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError,'frozen'):assert_fixed_reader(model,lora,digest)
        lora.requires_grad_(False)
        with torch.no_grad():next(lora.parameters()).add_(1)
        with self.assertRaisesRegex(RuntimeError,'changed'):assert_fixed_reader(model,lora,digest)

    def test_baseline_is_reused_for_both_methods_and_detects_changes(self):
        dev={'images':[{'image_id':'im','questions':[{'question_id':'q'}]}]}
        rows=[{'image_id':'im','question_id':'q','condition':c,'answer':'71','score':1.} for c in ('CLEAN','LR4')]
        result={'CLEAN':100.,'LR4':100.},rows
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);binding={'reader':'A2','view':'full'}
            with patch('scripts.s1_a2_convergence.evaluator'),patch('scripts.s1_a2_convergence.evaluate',return_value=result) as evaluate:
                self.assertEqual(baseline_once(None,None,None,dev,path,binding),result)
                self.assertEqual(baseline_once(None,None,None,dev,path,binding),result)
                self.assertEqual(evaluate.call_count,1)
            (path/'summary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                baseline_once(None,None,None,dev,path,binding)

    def test_study_requires_both_controls_and_fixed_a2(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/s1_a2_convergence.json').read_text())
        validate_study(config)
        for name,value in [('freeze_a2',False),('methods',['restoration']),('reader','A0')]:
            invalid=copy.deepcopy(config);invalid[name]=value
            with self.assertRaises(ValueError):validate_study(invalid)


if __name__=='__main__':unittest.main()
