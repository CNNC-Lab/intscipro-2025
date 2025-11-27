"""
Spike analysis module for Exercise 4.
Contains functions to be tested with pytest.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional


def calculate_firing_rate(spike_times: NDArray, duration: float) -> float:
    """Calculate firing rate in Hz."""
    if duration <= 0:
        raise ValueError("Duration must be positive")
    return len(spike_times) / duration


def calculate_cv(spike_times: NDArray) -> float:
    """Calculate coefficient of variation of ISI.
    
    CV = std(ISI) / mean(ISI)
    """
    if len(spike_times) < 2:
        raise ValueError("Need at least 2 spikes")
    
    isis = np.diff(spike_times)
    if np.mean(isis) == 0:
        return np.inf
    
    return np.std(isis) / np.mean(isis)


def bin_spike_train(spike_times: NDArray, 
                   bin_size: float,
                   duration: float) -> NDArray:
    """Bin spike train into time windows.
    
    Parameters
    ----------
    spike_times : NDArray
        Spike times in seconds
    bin_size : float
        Bin size in seconds
    duration : float
        Total duration in seconds
        
    Returns
    -------
    NDArray
        Spike counts per bin
    """
    if bin_size <= 0:
        raise ValueError("Bin size must be positive")
    
    n_bins = int(np.ceil(duration / bin_size))
    counts, _ = np.histogram(spike_times, bins=n_bins, range=(0, duration))
    return counts


def detect_bursts(spike_times: NDArray,
                 max_isi: float = 0.01,
                 min_spikes: int = 3) -> list:
    """Detect bursts in spike train.
    
    A burst is defined as min_spikes consecutive spikes with
    ISI < max_isi.
    
    Parameters
    ----------
    spike_times : NDArray
        Spike times in seconds
    max_isi : float
        Maximum ISI within burst (default: 0.01s = 10ms)
    min_spikes : int
        Minimum spikes to constitute a burst (default: 3)
        
    Returns
    -------
    list
        List of tuples (burst_start_idx, burst_end_idx)
    """
    if len(spike_times) < min_spikes:
        return []
    
    isis = np.diff(spike_times)
    bursts = []
    burst_start = None
    burst_count = 1
    
    for i, isi in enumerate(isis):
        if isi < max_isi:
            if burst_start is None:
                burst_start = i
            burst_count += 1
        else:
            if burst_start is not None and burst_count >= min_spikes:
                bursts.append((burst_start, i))
            burst_start = None
            burst_count = 1
    
    # Check if we ended in a burst
    if burst_start is not None and burst_count >= min_spikes:
        bursts.append((burst_start, len(spike_times) - 1))
    
    return bursts
