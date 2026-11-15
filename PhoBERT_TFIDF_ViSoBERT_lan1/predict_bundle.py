"""Checkpoint inference only. Never fits or changes thresholds using test text."""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from pathlib import Path
import hashlib
import json
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from pyvi import ViTokenizer

ASPECTS = ['nhan_vien', 'sach_se', 'vi_tri', 'phong', 'gia_tri', 'an_uong', 'tien_nghi']
LABELS = [f'{a}_{s}' for a in ASPECTS for s in ['nhac', 'khen', 'che']]


def basic_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', str(text))).strip()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class JointModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        config = AutoConfig.from_pretrained(cfg['model_name'], revision=cfg['revision'],
                                            trust_remote_code=False)
        # All trained weights come from the supplied checkpoint, not a fresh encoder.
        self.encoder = AutoModel.from_config(config, add_pooling_layer=False,
                                             trust_remote_code=False)
        hidden = config.hidden_size
        dropout = cfg.get('dropout', .2)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, hidden),
            nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, len(LABELS)))

    def forward(self, inputs):
        return self.head(self.encoder(**inputs).last_hidden_state[:, 0])


class Predictor:
    def __init__(self, model_dir, batch_size=8, device=None):
        self.folder = Path(model_dir).resolve()
        self.cfg = json.loads((self.folder / 'config.json').read_text())
        self.batch_size = batch_size
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.kind = self.cfg.get('architecture', 'joint_21')
        self.sources = None
        self.model = None
        if self.cfg.get('columns') != LABELS:
            raise ValueError('Checkpoint has different label columns/order')
        if self.kind == 'three_model_probability_blend':
            self.sources = [Predictor(self.folder / p, batch_size, str(self.device))
                            for p in self.cfg['sources']]
            weights = np.asarray(self.cfg['weights'], dtype=float)
            assert len(weights) == 3 and np.all(weights >= .1 - 1e-10)
            assert abs(weights.sum() - 1) < 1e-10
        elif self.kind == 'tfidf_dual_text':
            import joblib
            self.model = joblib.load(self.folder / 'model.joblib')
        elif self.kind == 'joint_21' and self.cfg.get('head', 'joint') == 'joint':
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg['model_name'], revision=self.cfg['revision'],
                use_fast=self.cfg.get('fast_tokenizer', False), trust_remote_code=False)
            self.model = JointModel(self.cfg)
            checkpoint = self.folder / self.cfg.get('checkpoint_file', 'model.pt')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state, strict=True)
            del state
            self.model.to(self.device).eval()
        else:
            raise ValueError('Unsupported checkpoint architecture: ' + self.kind)

    @torch.inference_mode()
    def probabilities(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, 21))
        if self.sources is not None:
            return sum(w * s.probabilities(texts)
                       for w, s in zip(self.cfg['weights'], self.sources))
        if self.kind == 'tfidf_dual_text':
            from colab_runner import text_views, transform_views
            views = text_views(texts)
            x = transform_views(self.model['vectorizers'], views, range(len(texts)))
            return self.model['classifier'].predict_proba(x)
        if self.cfg.get('expand_abbreviations', False):
            from colab_runner import normalize_text
            cleaner = normalize_text
        elif self.cfg.get('preprocessing_mode') == 'basic_social':
            from colab_runner import basic_text as social_text
            cleaner = social_text
        else:
            cleaner = basic_text
        width = self.cfg.get('max_length', 256) - self.tokenizer.num_special_tokens_to_add(pair=False)
        stride = self.cfg.get('stride', 64)
        assert 0 <= stride < width
        windows, owners = [], []
        for i, text in enumerate(texts):
            text = cleaner(text)
            if self.cfg.get('segment', True):
                text = ViTokenizer.tokenize(text)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, max(1, len(ids)), width - stride):
                seq = self.tokenizer.build_inputs_with_special_tokens(ids[start:start + width])
                windows.append({'input_ids': seq, 'attention_mask': [1] * len(seq)})
                owners.append(i)
                if start + width >= len(ids):
                    break
        logits = np.full((len(texts), 21), -np.inf, dtype=np.float32)
        owners = np.asarray(owners)
        for start in range(0, len(windows), self.batch_size):
            inputs = self.tokenizer.pad(windows[start:start + self.batch_size],
                                        padding=True, return_tensors='pt')
            result = self.model({k: v.to(self.device) for k, v in inputs.items()})
            # Same review-level max pooling as training, with bounded GPU memory.
            np.maximum.at(logits, owners[start:start + len(result)], result.float().cpu().numpy())
            if start == 0 or (start // self.batch_size + 1) % 25 == 0:
                print(f'Đã dự đoán {min(start + self.batch_size, len(windows))}/{len(windows)} cửa sổ', flush=True)
        assert np.isfinite(logits).all()
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    def frame(self, ids, probabilities):
        assert self.cfg.get('threshold_space', 'probability') == 'probability'
        thresholds = np.asarray(self.cfg['thresholds'], dtype=float)
        assert thresholds.shape == (21,) and np.isfinite(thresholds).all()
        assert np.all((thresholds > 0) & (thresholds < 1))
        prediction = (probabilities >= thresholds).astype(np.int8)
        for j in range(0, 21, 3):
            prediction[:, j] |= prediction[:, j + 1] | prediction[:, j + 2]
        frame = pd.DataFrame(prediction, columns=LABELS)
        frame.insert(0, 'id', np.asarray(ids, dtype=str))
        return frame


def write_submission(engine, test_path, demo_path, output):
    test = pd.read_csv(test_path, dtype={'id': str}, keep_default_na=False)
    demo = pd.read_csv(demo_path, dtype=str, keep_default_na=False)
    assert list(test.columns) == ['id', 'text'], 'Test must contain id,text only'
    assert list(demo.columns) == ['id'] + LABELS, 'Template column order differs'
    assert len(test) == len(demo) == 732, 'Official submission requires 732 data rows'
    assert test.id.is_unique and demo.id.is_unique
    assert test.id.ne('').all() and test.text.ne('').all()
    assert set(test.id) == set(demo.id), 'Test IDs do not match the official demo'
    scores = engine.probabilities(test.text)
    frame = engine.frame(test.id, scores)
    frame = frame.set_index('id').loc[demo.id].reset_index()
    frame.to_csv(output, index=False, encoding='utf-8', lineterminator='\n')
    saved = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert list(saved.columns) == list(demo.columns)
    assert saved.id.tolist() == demo.id.tolist() and saved.id.is_unique
    assert saved[LABELS].isin(['0', '1']).all().all()
    import score_absa
    assert score_absa.kiem_tra(output, test_path) == 0
    print('Submission SHA256:', file_hash(output), flush=True)
    return frame
