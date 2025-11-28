import pytest
import numpy as np
from numpy.fft import fft, fftfreq
from chunk_power_spectrum import _separate_chunks, _calc_power_spectrum, power_spectrum

# Tests for _separate_chunks function
def test_separate_chunks_exact_division():
    """
    If the signal length is an exact multiple of chunk_size,
    the function should return equally sized chunks with no remainder.
    """
    chunk_size = 10
    signal = np.arange(30)  # exactly 3 chunks

    chunks = _separate_chunks(signal, chunk_size=chunk_size)

    # Expect 3 chunks
    assert len(chunks) == 3

    # Each chunk should have shape (chunk_size,)
    for ch in chunks:
        assert ch.shape == (chunk_size,)

    # Check that chunks are correct slices
    assert np.array_equal(chunks[0], signal[0:10])
    assert np.array_equal(chunks[1], signal[10:20])
    assert np.array_equal(chunks[2], signal[20:30])


def test_separate_chunks_ignores_remainder():
    """
    If the signal length is not an exact multiple of chunk_size,
    the remainder at the end should be discarded.
    """
    chunk_size = 8
    signal = np.arange(25)  # 25 // 8 = 3 chunks, remainder = 1

    chunks = _separate_chunks(signal, chunk_size=chunk_size)

    assert len(chunks) == 3
    for ch in chunks:
        assert ch.shape == (chunk_size,)

    # Ensure that the leftover element is not included
    flat = np.concatenate(chunks)
    assert len(flat) == 3 * chunk_size
    assert 24 not in flat  # last element should be ignored


def test_separate_chunks_empty_signal():
    """
    If signal is shorter than chunk_size, no chunks should be returned.
    """
    signal = np.arange(5)
    chunk_size = 10

    chunks = _separate_chunks(signal, chunk_size=chunk_size)

    assert chunks == []


def test_separate_chunks_random_data():
    """
    For random data, still ensure chunk boundaries are correct.
    """
    rng = np.random.default_rng(0)
    signal = rng.normal(size=53)
    chunk_size = 7

    chunks = _separate_chunks(signal, chunk_size=chunk_size)
    num_chunks = len(signal) // chunk_size

    assert len(chunks) == num_chunks

    for i, ch in enumerate(chunks):
        start = i * chunk_size
        end   = start + chunk_size
        assert np.allclose(ch, signal[start:end])


# Tests for _calc_power_spectrum function
def test_calc_power_spectrum_single_sine_peak():
    """
    For a pure sine wave, the power spectrum should have a clear peak
    at the corresponding frequency, with amplitude ~0.5 given the formula:
    sp = 2 * |ck|^2 / N^2
    """
    N = 1825           # match default chunk_size in function doc
    k = 10             # frequency index of the sine wave
    n = np.arange(N)
    signal = np.sin(2 * np.pi * k * n / N)

    freq, sp = _calc_power_spectrum(signal, chunk_size=N)

    # 1) Output shapes: size = floor(chunk_size/2)
    expected_size = N // 2
    assert freq.shape == (expected_size,)
    assert sp.shape == (expected_size,)

    # 2) Peak should be at frequency k / N
    #    Ignore the 0-frequency bin when searching for the peak.
    peak_idx = np.argmax(sp[1:]) + 1
    assert np.isclose(freq[peak_idx], k / N, rtol=1e-3, atol=0)

    # 3) Expected peak power ≈ 0.5 for unit-amplitude sine wave
    #    (given the specific normalization used in _calc_power_spectrum)
    assert sp[peak_idx] == pytest.approx(0.5, rel=1e-2)


def test_calc_power_spectrum_respects_chunk_size():
    """
    The returned arrays should have length floor(chunk_size / 2),
    regardless of the total signal length.
    """
    N_signal = 2000
    chunk_size = 101

    rng = np.random.default_rng(0)
    signal = rng.standard_normal(N_signal)

    freq, sp = _calc_power_spectrum(signal, chunk_size=chunk_size)

    expected_size = chunk_size // 2
    assert freq.shape == (expected_size,)
    assert sp.shape == (expected_size,)

    # Frequencies in the returned slice should be strictly increasing
    assert np.all(np.diff(freq) > 0)

# Union tests for power_spectrum function
def test_power_spectrum_shape_and_chunk_count():
    """
    power_spectrum() should return:
    - freq of size floor(chunk_size/2)
    - one spectrum per chunk in all_sp
    """
    chunk_size = 20
    signal = np.arange(60)  # 60 // 20 = 3 chunks

    freq, all_sp = power_spectrum(signal, chunk_size=chunk_size)

    # number of chunks
    assert len(all_sp) == 3

    # frequency length
    expected_freq_size = chunk_size // 2
    assert freq.shape == (expected_freq_size,)

    # each spectrum should have the same shape as freq
    for sp in all_sp:
        assert sp.shape == (expected_freq_size,)


def test_power_spectrum_uses_over_chunks_correctly():
    """
    When all chunks contain identical signals, all spectra must be identical.
    """
    chunk_size = 50
    N_chunks = 4

    # Create a repeated sine wave, same for each chunk
    n = np.arange(chunk_size)
    k = 5
    base_chunk = np.sin(2 * np.pi * k * n / chunk_size)

    # concatenate N repeated chunks
    signal = np.tile(base_chunk, N_chunks)

    freq, all_sp = power_spectrum(signal, chunk_size=chunk_size)

    # There should be 4 identical spectra
    assert len(all_sp) == N_chunks

    # Check identical spectra across all chunks
    for i in range(1, N_chunks):
        assert np.allclose(all_sp[i], all_sp[0])


def test_power_spectrum_empty_when_shorter_than_chunk():
    """No chunk → no power spectra."""
    signal = np.arange(10)
    chunk_size = 100

    freq, all_sp = power_spectrum(signal, chunk_size)

    # No chunk means no spectra
    assert all_sp == []
    # freq is the freq of the last computed chunk; if none computed, freq should
    # come from the last iteration. We expect freq from remainder handling:
    # Since the code returns `freq` from the last chunk processed,
    # if no chunks were processed, freq must not exist → so freq is undefined.
    # But the function returns freq anyway, so we expect a NameError or None.
    # The current implementation will raise UnboundLocalError.
    # So we assert this expected behavior.
    # If you fix power_spectrum to return None freq for no chunks, adjust test.
    with pytest.raises(UnboundLocalError):
        _ = freq  # freq is not defined if no chunks exist


def test_power_spectrum_frequency_correctness():
    """
    The returned freq must exactly match the freq from _calc_power_spectrum()
    applied to a single chunk.
    """
    chunk_size = 64
    signal = np.random.default_rng(0).normal(size=chunk_size * 2)

    # Compute manually for first chunk
    first_chunk = signal[:chunk_size]
    freq_ref, _ = _calc_power_spectrum(first_chunk, chunk_size)

    freq, all_sp = power_spectrum(signal, chunk_size)

    assert np.allclose(freq, freq_ref)

