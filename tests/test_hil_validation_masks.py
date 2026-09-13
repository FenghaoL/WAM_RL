import unittest
import torch
from ahawam.trainer import Wan22Trainer

class ValidationMaskTests(unittest.TestCase):
    def test_human_boundary_masks_survive_eval_batching(self):
        sample=dict(video=torch.zeros(3,13,2,2), prompt='test', action=torch.zeros(48,6),
                    action_is_pad=torch.arange(48)>=7, image_is_pad=torch.arange(13)>=2,
                    proprio_is_pad=torch.arange(49)>=8, video_supervision_mask=torch.tensor(False),
                    data_source=torch.tensor(1))
        batch=Wan22Trainer._to_batched_eval_sample(sample)
        for key in ('action_is_pad','image_is_pad','proprio_is_pad'):
            self.assertTrue(torch.equal(batch[key],sample[key].unsqueeze(0)))
        self.assertEqual(batch['video_supervision_mask'].tolist(),[False])
        self.assertEqual(batch['data_source'].tolist(),[1])

if __name__=='__main__': unittest.main()
