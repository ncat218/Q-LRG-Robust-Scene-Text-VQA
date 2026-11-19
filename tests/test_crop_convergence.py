import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from qlrg.common import object_hash, write_json
from scripts.convergence_policy import Selection
from scripts.crop_matrix import arm_name, comparison_pairs, specifications, view_plan
from scripts.s1_convergence import Features, merged_train, validate_config


class CropConvergenceTests(unittest.TestCase):
    def test_matrix_is_32_unique_arms_and_comparisons_are_within_budget(self):
        names={arm_name(*s) for s in specifications()}
        self.assertEqual(len(names),32)
        for left,right in comparison_pairs(('A0','A2')):
            self.assertIn(left,names);self.assertIn(right,names)
            self.assertEqual(left.split('__')[1],right.split('__')[1])

    def test_fair_views_sum_to_576_and_restore_flags_follow_each_view(self):
        im=Image.new('RGB',(280,140))
        for _,budget,layout,flags in specifications(('A0',)):
            views,names,active,tokens=view_plan(im,(28,14,140,112),layout,budget,flags)
            self.assertEqual(active,flags)
            if budget=='fair576':
                self.assertEqual(sum(v.width*v.height//(28*28) for v in views),576)
                self.assertEqual(tokens,288 if layout=='fusion' else 576)

    def test_no_roi_fusion_does_not_restore_full_when_only_crop_was_requested(self):
        im=Image.new('RGB',(280,140))
        _,names,active,_=view_plan(im,None,'fusion','fair576',(False,True))
        self.assertEqual(names,('full',));self.assertEqual(active,(False,))
        _,names,active,_=view_plan(im,None,'crop','native',(True,))
        self.assertEqual(names,('full',));self.assertEqual(active,(True,))

    def test_clean_constraint_and_patience_do_not_select_a_high_lr4_unsafe_model(self):
        policy=Selection(80,best_lr4=60,progress_lr4=60)
        result=policy.observe(65,78,100,0.5)
        self.assertFalse(result['eligible']);self.assertFalse(result['save_best'])
        self.assertEqual(policy.best_lr4,60)
        result=policy.observe(61,79.1,200,1.)
        self.assertTrue(result['save_best'])
        for step in range(4):result=policy.observe(61.02,80,300+step*100,1.5+step*.5)
        self.assertTrue(result['stop'])
        # Absolute best and the min-delta patience reference are deliberately distinct.
        self.assertEqual(policy.best_lr4,61.02)
        self.assertEqual(Selection(**policy.state_dict()).state_dict(),policy.state_dict())

    def test_official_train_merge_keeps_official_val_out_and_checks_content_overlap(self):
        def m(split,iid):return {'split':split,'images':[{'image_id':iid,'image_sha256':iid,
                                                        'questions':[{'question_id':iid}]}]}
        manifests={s:m(s,i) for s,i in [('train_inner','a'),('val_inner','b'),('official_val','c')]}
        self.assertEqual([im['image_id'] for im in merged_train(manifests)['images']],['a','b'])
        manifests['official_val']['images'][0]['image_sha256']='a'
        with self.assertRaises(ValueError):merged_train(manifests)

    def test_teacher_and_student_use_same_lr4_selected_crop_and_query(self):
        class Model:
            def __init__(self):self.seen=[]
            def prepare(self,im,question):
                self.seen.append((im.size,question))
                return {'image_grid_thw':torch.tensor([[1,8,8]]),'input_ids':torch.tensor([[1,2]]),
                        'attention_mask':torch.tensor([[1,1]])}
            def encode(self,batch):return torch.zeros(16,4)
        clean=Image.new('RGB',(280,140),'white');lr4=Image.new('RGB',clean.size,'gray')
        questions=[{'question_id':'q','question':'Who?'}]
        image={'image_id':'im','questions':questions}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=Model()
            write_json(root/'proposals'/f"{object_hash('im')[:24]}.json",
                       {'image_id':'im','lr4_sha256':hashlib.sha256(lr4.tobytes()).hexdigest(),
                        'questions':{'q':{'box':[28,14,140,112]}}})
            cache=Features(model,root,'crop',root)
            with patch('scripts.s1_convergence.paired_images',return_value=(clean,lr4)):
                cache.load(image,questions[0])
            self.assertEqual(model.seen[0],model.seen[1])
            self.assertEqual(model.seen[0][1],'Who?')
            self.assertEqual(cache.images['CLEAN'].size,cache.images['LR4'].size)
            self.assertEqual(cache.features['CLEAN'].shape,cache.features['LR4'].shape)


if __name__=='__main__':unittest.main()
