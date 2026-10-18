import numpy as np
import pytest
from src.data import signal_features, spectral_features, load_profile, load_sensor, build_feature_frame, SAMPLE_RATES


def test_statistics_and_constants():
    f = signal_features([1, 2, 3, 4])
    assert f['mean'] == 2.5
    assert f['energy'] == pytest.approx(f['rms'] ** 2)
    assert f['ptp'] == 3
    assert np.isfinite(list(signal_features(np.zeros(60)).values())).all()
    for bad in [[], [1], [1, np.nan], [[1, 2]]]:
        with pytest.raises(ValueError):
            signal_features(bad)


def test_fft_sinusoid_offset_invariance():
    x = np.sin(2 * np.pi * 10 * np.arange(6000) / 100)
    f = spectral_features(x)
    assert f['fft_peak_hz'] == pytest.approx(10)
    assert f['fft_band_5_20'] > .999
    assert sum(v for k, v in f.items() if 'band' in k) == pytest.approx(1)
    assert spectral_features(x + 20) == pytest.approx(f)
    assert all(v == 0 for v in spectral_features(np.ones(6000)).values())
    assert sum(60 * rate for rate in SAMPLE_RATES.values()) == 43680


def test_loading_alignment_and_invalid_data(tmp_path):
    np.savetxt(tmp_path / 'profile.txt', [[3, 100, 0, 130, 0], [20, 90, 1, 115, 1]], delimiter='\t')
    np.savetxt(tmp_path / 'TS1.txt', np.ones((2, 60)), delimiter='\t')
    assert load_profile(tmp_path).shape == (2, 5)
    assert build_feature_frame(['TS1'], False, tmp_path).shape == (2, 11)
    with pytest.raises(ValueError):
        load_sensor('../profile', tmp_path)
    np.savetxt(tmp_path / 'TS1.txt', np.ones((1, 60)), delimiter='\t')
    with pytest.raises(ValueError, match='Cycle count'):
        build_feature_frame(['TS1'], False, tmp_path)
    np.savetxt(tmp_path / 'TS1.txt', np.ones((2, 59)), delimiter='\t')
    with pytest.raises(ValueError, match='Invalid samples'):
        load_sensor('TS1', tmp_path)
    np.savetxt(tmp_path / 'profile.txt', [[3, 100, 0, 130, .5]], delimiter='\t')
    with pytest.raises(ValueError):
        load_profile(tmp_path)
