"""Regression coverage for failures found during the deadline readiness audit."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.artifacts import atomic_path
from src.data.cache import build_image_cache, open_image_cache
from src.data.crop import crop_and_relabel_masks
from src.data.config import SegConfig
from src.data.mask_cache import save_mask, load_mask
from src.training.common import score_by_id, pred_artifact_path
from src.training.calibration import _fit_ridge
from src.training.regression_cnn import SmallRegressionCNN, RegressionConfig, train_regression_cnn, MetricRegressionDataset, _run_epoch
from src.utils.evaluate import calculate_metrics_from_masks
from src.training.common import write_manifest


class AuditTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'ID':['a','b'], 'CD':[100.,200.], 'CV':[.2,.3], 'HEX':[.4,.5]})

    def test_scoring_requires_requested_coverage(self):
        gt = self.frame()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            score_by_id(gt.iloc[:1], gt, expected_ids=gt.ID)

    def test_scoring_rejects_duplicates_and_nan(self):
        gt = self.frame()
        for pred in (pd.concat([gt,gt]), gt.assign(CD=np.nan), gt.assign(CV=np.inf)):
            with self.assertRaises(ValueError):
                score_by_id(pred, gt)

    def test_missing_id_is_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            score_by_id(self.frame().drop(columns='ID'), self.frame())

    def test_manifest_parallel_writes_preserve_every_method(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as d:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: write_manifest(Path(d),method=str(i),metrics={'score':i}),range(12)))
            self.assertEqual(len(json.loads((Path(d)/'manifest.json').read_text())['methods']),12)

    def test_mha_parser_respects_big_endian_values(self):
        from src.io_utils import _load_mha
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'big.mha'
            header=b'DimSize = 2 2\nElementType = MET_USHORT\nBinaryDataByteOrderMSB = True\nElementDataFile = LOCAL\n'
            values=np.array([[1,256],[500,1000]],dtype='>u2')
            path.write_bytes(header+values.tobytes())
            np.testing.assert_array_equal(_load_mha(path),values)

    def test_evaluation_default_follows_split(self):
        from evaluate import build_parser,resolve_paths
        args=resolve_paths(build_parser().parse_args(['--split','val']))
        self.assertTrue(args.predictions_csv.endswith('predictions_val.csv'))

    def test_bundle_rejects_changed_checkpoint(self):
        from inference import predict_model_bundle
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'submission.json').write_text(json.dumps({'method':'ensemble','members':[{'file':'member.pt','weight':1,'sha256':'wrong'}]}))
            (root/'member.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                predict_model_bundle(root/'image.mha',root)

    def test_submission_json_rejects_nan(self):
        from inference import write_json_file
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                write_json_file(location=Path(d)/'out.json',content=float('nan'))

    def test_partial_artifact_cannot_replace_full_split(self):
        self.assertNotEqual(pred_artifact_path(Path('/tmp'), 'hash', 'val'),
                            pred_artifact_path(Path('/tmp'), 'hash', 'val', limit=10))

    def test_noncontiguous_labels_have_same_metrics(self):
        sparse = np.zeros((16,16), dtype=np.int32)
        sparse[1:5,1:5] = 2
        sparse[8:14,8:14] = 20
        dense = crop_and_relabel_masks(sparse,(0,0,16,16))
        self.assertEqual(calculate_metrics_from_masks(sparse), calculate_metrics_from_masks(dense))
        self.assertEqual(calculate_metrics_from_masks(sparse)['Number of Cells'],2)

    def test_crop_relabel_matches_legacy_with_and_without_background(self):
        for masks in (np.array([[4,4],[8,8]],dtype=np.int32),
                      np.array([[0,4],[8,0]],dtype=np.int32),
                      np.zeros((2,2), dtype=np.int32)):
            old = np.zeros_like(masks)
            for label, value in enumerate(np.unique(masks[masks>0]),start=1):
                old[masks==value]=label
            np.testing.assert_array_equal(crop_and_relabel_masks(masks,(0,0,2,2)),old)

    def test_atomic_failure_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'artifact.json'; path.write_text('old')
            with self.assertRaises(RuntimeError):
                with atomic_path(path) as temporary:
                    temporary.write_text('partial')
                    raise RuntimeError('interrupted')
            self.assertEqual(path.read_text(),'old')
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_mask_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            mask=np.array([[0,1],[2,0]],dtype=np.uint16)
            save_mask(Path(d),'hash',0,mask)
            np.testing.assert_array_equal(load_mask(Path(d),'hash',0),mask)

    def test_ridge_without_intercept_penalizes_first_feature(self):
        x=np.ones((2,1)); y=np.ones(2)
        np.testing.assert_allclose(_fit_ridge(x,y,alpha=2,fit_intercept=False),[.5])

    def test_overnight_batch_hash_matches_worker_cli(self):
        from scripts.run_overnight import OvernightRunner, build_parser, _seg_config_cli
        args=build_parser().parse_args(['--batch_size','32'])
        runner=OvernightRunner(args)
        for _, cfg in [('default',runner.default_seg), *runner.ablation_configs]:
            self.assertEqual(cfg.batch_size,32)
            argv=_seg_config_cli(cfg,32)
            self.assertEqual(argv[argv.index('--batch_size')+1],'32')
        with self.assertRaises(ValueError):
            _seg_config_cli(SegConfig(batch_size=16),32)

    def test_uint8_model_preprocessing_matches_legacy(self):
        torch.set_num_threads(1)
        model=SmallRegressionCNN().eval()
        image=torch.randint(0,256,(2,1,256,256),dtype=torch.uint8)
        with torch.no_grad():
            torch.testing.assert_close(model(image),model(image.float()/255),rtol=0,atol=0)

    def test_epoch_scoring_uses_exact_zero_targets_with_shuffled_loader(self):
        from torch.utils.data import DataLoader
        from src.training.common import target_stats
        frame=pd.DataFrame({'idx':[0,1,2],'ID':['a','b','c'],
                            'CD':[100.,110.,120.], 'CV':[.3,.4,.5], 'HEX':[0.,.53,.55]})
        stats=target_stats(frame)
        ds=MetricRegressionDataset(np.zeros((3,64,64),dtype=np.uint8),frame,stats)
        class Constant(torch.nn.Module):
            def forward(self,x):
                return torch.zeros((len(x),3))
        loss, scores=_run_epoch(Constant(),DataLoader(ds,batch_size=2,shuffle=True),
                               torch.nn.HuberLoss(),torch.device('cpu'),stats)
        pred=frame[['ID']].copy()
        for c in ['CD','CV','HEX']: pred[c]=stats[c]['mean']
        expected=score_by_id(pred,frame)
        for c in expected: self.assertAlmostEqual(scores[c],expected[c],places=4)

    def make_cache(self, root):
        data=root/'images'; data.mkdir(); cache=root/'cache'
        labels=pd.DataFrame({'ID':[f'image{i}' for i in range(20)],
                             'slide_id':[f'slide{i}' for i in range(20)],
                             'CD':np.arange(20)+100.,'CV':.3,'HEX':.5})
        for name in labels.ID: (data/f'{name}.mha').touch()
        labels_csv=root/'labels.csv'; labels.to_csv(labels_csv,index=False)
        with patch('src.data.cache._decode_gray_mha',return_value=np.zeros((4,4),dtype=np.uint8)):
            build_image_cache(data,labels_csv,cache,expected_count=20)
        return data,labels_csv,cache

    @patch('src.data.cache.EXPECTED_SHAPE',(4,4))
    def test_cache_limit_cannot_destroy_existing_cache_or_reset_splits(self):
        with tempfile.TemporaryDirectory() as d:
            data,labels,cache=self.make_cache(Path(d))
            before=(cache/'splits.json').read_bytes()
            build_image_cache(data,labels,cache,expected_count=20,seed=99)
            self.assertEqual((cache/'splits.json').read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'does not match'):
                build_image_cache(data,labels,cache,limit=10)
            self.assertEqual(len(open_image_cache(cache)[1]),20)
            (cache/'masks').mkdir()
            with self.assertRaisesRegex(ValueError,'derived masks'):
                build_image_cache(data,labels,cache,force=True,expected_count=20)

    @patch('src.data.cache.EXPECTED_SHAPE',(4,4))
    def test_cache_truncation_detected(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,cache=self.make_cache(Path(d))
            (cache/'images_u8.npy').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'byte size'):
                open_image_cache(cache)

    @patch('src.data.cache.EXPECTED_SHAPE',(4,4))
    def test_best_checkpoint_survives_interrupted_next_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); _,labels,cache=self.make_cache(root)
            scores={'CD':10.,'CV':10.,'HEX':10.,'mean':10.}
            with patch('src.training.regression_cnn._run_epoch',
                       side_effect=[(1.,scores),(1.,scores),RuntimeError('interrupt')]), \
                 patch('torch.cuda.is_available',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'interrupt'):
                    train_regression_cnn(cache,labels,root/'run',
                        config=RegressionConfig(epochs=2,num_workers=0))
            ckpt=torch.load(root/'run'/'best_model.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ckpt['epoch'],1)
            self.assertEqual(len(pd.read_csv(root/'run'/'history.csv')),1)


if __name__=='__main__':
    unittest.main()
