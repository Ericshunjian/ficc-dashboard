"""Fit real BART using StochTree's compiled classic MCMC sampler.

No test labels are supplied. Residual variance is fixed from the training Ridge
fit (empirical Bayes); likelihood variance is six times that value, equivalent
to a 1/6 power likelihood for the tree posterior conditional on fixed noise.
Prediction noise uses the original training estimate, not the tempered variance.
"""
from __future__ import annotations

import gzip
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'output' / 'bayes' / 'train_202512'
SEED, CHAINS, DRAWS, BURNIN = 20261020, 4, 1500, 1500


def main(cache=CACHE, train_through='2025-12-31'):
    import arviz as az
    from stochtree import BARTModel
    cache = Path(cache).resolve()
    if not cache.is_relative_to((ROOT / 'output' / 'bayes').resolve()):
        raise ValueError('BART cache must remain inside output/bayes')
    source = cache / 'bart_inputs.npz'
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    summary_path = cache / 'bart_summary.json'
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding='utf-8'))
        if old['inputs_sha256'] != checksum:
            raise ValueError('Existing BART inputs changed; use a new experiment version')
        print('Frozen BART cache already exists; no retraining.')
        return
    data = np.load(source)
    metadata = json.loads((cache / 'study_inputs.json').read_text(encoding='utf-8'))
    x, y, score = data['x_train'], data['y_train'], data['x_score']
    if x.shape != (metadata['train_n'], 28) or metadata['train_through'] != train_through:
        raise ValueError('Unexpected training population')
    if np.any(data['train_label_exit_dates'] > train_through):
        raise ValueError('A training label crosses the requested cutoff')
    noise = metadata['noise_variance']
    print(f'Classic MCMC BART: {len(y)} training labels, 28 features; no test Y.', flush=True)
    model = BARTModel()
    model.sample(x, y, X_test=score, num_gfr=0, num_burnin=BURNIN, num_mcmc=DRAWS,
                 general_params={'random_seed': SEED, 'num_chains': CHAINS, 'num_threads': 1,
                                 'standardize': True, 'sample_sigma2_global': False,
                                 'sigma2_init': 6 * noise / np.var(y), 'keep_burnin': False, 'keep_gfr': False},
                 mean_forest_params={'num_trees': 50, 'alpha': .95, 'beta': 2.,
                                     'sample_sigma2_leaf': False, 'sigma2_leaf_init': .25 / 50,
                                     'min_samples_leaf': 5, 'max_depth': 6})
    # Samples are consecutive by chain. Evaluate each retained ensemble directly.
    latent = np.asarray(model.extract_parameter('y_hat_test')).T
    fitted = np.asarray(model.extract_parameter('y_hat_train')).T
    if latent.shape != (CHAINS * DRAWS, len(score)):
        raise ValueError(f'Unexpected posterior shape {latent.shape}')
    serialized = model.to_json()
    with gzip.open(cache / 'bart_model.json.gz', 'wt', encoding='utf-8') as handle:
        handle.write(serialized)
    restored = BARTModel()
    restored.from_json(serialized)
    np.testing.assert_allclose(np.asarray(restored.predict(score, terms='y_hat')), latent.T, atol=1e-10)
    rng = np.random.default_rng(SEED + 2)
    predictive = latent + rng.normal(size=latent.shape) * np.sqrt(noise)
    std = np.sqrt(np.var(latent, axis=0) + noise)
    chain_mean = latent.reshape(CHAINS, DRAWS, -1).mean(axis=1)
    np.savez_compressed(cache / 'bart_predictions.npz', mean=latent.mean(axis=0), std=std,
                        low80=np.quantile(predictive, .1, axis=0), high80=np.quantile(predictive, .9, axis=0),
                        probability_positive=ndtr(latent / np.sqrt(noise)).mean(axis=0), chain_mean=chain_mean)
    idata = az.from_dict(posterior={'mu': fitted.reshape(CHAINS, DRAWS, -1)})
    rhat = np.asarray(az.rhat(idata, var_names=['mu'])['mu'])
    ess = np.asarray(az.ess(idata, var_names=['mu'])['mu'])
    summary = {'inputs_sha256': checksum, 'engine': 'StochTree classic BART MCMC',
               'engine_version': importlib.metadata.version('stochtree'),
               'random_seed': SEED, 'm': 50, 'chains': CHAINS, 'tune': BURNIN, 'draws_per_chain': DRAWS,
               'likelihood_power': 1 / 6, 'train_n': len(y), 'features_n': 28, 'train_through': train_through,
               'noise_variance_fixed': noise, 'sigma_diagnostics': {'r_hat': None, 'ess_bulk': None},
               'latent_rhat_median': float(np.median(rhat)), 'latent_rhat_p95': float(np.quantile(rhat, .95)),
               'latent_rhat_max': float(np.max(rhat)), 'latent_ess_median': float(np.median(ess)),
               'latent_ess_p05': float(np.quantile(ess, .05)),
               'test_chain_mean_max_range': float(np.max(np.ptp(chain_mean[:, 250:], axis=0))),
               'uncertainty_caveat': '噪声方差由训练岭回归残差估计并固定；条件于该估计，用1/6次幂似然近似处理重叠标签。未完整建模残差自相关及噪声参数不确定性，区间和概率须检验校准。'}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--train-through', default='2025-12-31')
    args = parser.parse_args()
    main(args.cache, args.train_through)
