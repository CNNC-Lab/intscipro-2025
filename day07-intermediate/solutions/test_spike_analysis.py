"""
Exercise 4: Unit Testing with pytest
Complete test suite for spike_analysis module.
"""

import pytest
import numpy as np
from spike_analysis import (
    calculate_firing_rate, 
    calculate_cv,
    bin_spike_train,
    detect_bursts
)


# Fixtures for reusable test data
@pytest.fixture
def regular_spikes():
    """Regular 10 Hz spiking."""
    return np.arange(0, 1.0, 0.1)


@pytest.fixture
def irregular_spikes():
    """Irregular spiking pattern."""
    return np.array([0.01, 0.05, 0.3, 0.32, 0.9])


@pytest.fixture
def burst_spikes():
    """Spike train with clear bursts."""
    return np.array([0.1, 0.105, 0.11, 0.115,  # Burst 1
                    0.5, 0.506, 0.512, 0.518,  # Burst 2
                    0.9])                        # Isolated spike


class TestFiringRate:
    """Tests for firing rate calculation."""
    
    def test_basic_rate(self):
        """Test basic firing rate calculation."""
        spikes = np.array([0.1, 0.2, 0.3, 0.4])
        rate = calculate_firing_rate(spikes, duration=1.0)
        assert rate == 4.0
    
    def test_empty_spikes(self):
        """Test with no spikes."""
        spikes = np.array([])
        rate = calculate_firing_rate(spikes, duration=1.0)
        assert rate == 0.0
    
    @pytest.mark.parametrize("n_spikes,duration,expected", [
        (10, 1.0, 10.0),
        (20, 2.0, 10.0),
        (5, 0.5, 10.0),
        (100, 10.0, 10.0),
    ])
    def test_rate_parametrized(self, n_spikes, duration, expected):
        """Test firing rate with multiple parameter combinations."""
        # Generate evenly spaced spikes
        spikes = np.linspace(0, duration, n_spikes, endpoint=False)
        rate = calculate_firing_rate(spikes, duration)
        assert rate == pytest.approx(expected)
    
    def test_negative_duration_raises(self):
        """Test that negative duration raises ValueError."""
        spikes = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="Duration must be positive"):
            calculate_firing_rate(spikes, duration=-1.0)
    
    def test_zero_duration_raises(self):
        """Test that zero duration raises ValueError."""
        spikes = np.array([0.1, 0.2])
        with pytest.raises(ValueError, match="Duration must be positive"):
            calculate_firing_rate(spikes, duration=0.0)


class TestCoefficientOfVariation:
    """Tests for CV calculation."""
    
    def test_regular_spiking_low_cv(self, regular_spikes):
        """Regular spiking should have CV ≈ 0."""
        cv = calculate_cv(regular_spikes)
        assert cv < 0.1  # Nearly regular
    
    def test_irregular_spiking_high_cv(self, irregular_spikes):
        """Irregular spiking should have CV > 0.5."""
        cv = calculate_cv(irregular_spikes)
        assert cv > 0.5
    
    def test_poisson_cv_around_one(self):
        """Poisson process should have CV ≈ 1."""
        np.random.seed(42)
        isis = np.random.exponential(scale=0.1, size=1000)
        spike_times = np.cumsum(isis)
        cv = calculate_cv(spike_times)
        # CV should be close to 1.0 for Poisson process
        assert cv == pytest.approx(1.0, abs=0.1)
    
    def test_insufficient_spikes_raises(self):
        """Test that < 2 spikes raises ValueError."""
        # Test with 0 spikes
        with pytest.raises(ValueError, match="Need at least 2 spikes"):
            calculate_cv(np.array([]))
        
        # Test with 1 spike
        with pytest.raises(ValueError, match="Need at least 2 spikes"):
            calculate_cv(np.array([0.5]))


class TestBinning:
    """Tests for spike train binning."""
    
    def test_bin_counts(self):
        """Test basic binning."""
        spikes = np.array([0.05, 0.15, 0.25, 0.35])
        counts = bin_spike_train(spikes, bin_size=0.1, duration=1.0)
        
        # Should have 10 bins (0-0.1, 0.1-0.2, ..., 0.9-1.0)
        assert len(counts) == 10
        
        # Check specific bins
        assert counts[0] == 1  # 0.05 in first bin
        assert counts[1] == 1  # 0.15 in second bin
        assert counts[2] == 1  # 0.25 in third bin
        assert counts[3] == 1  # 0.35 in fourth bin
    
    def test_bin_size_invalid_raises(self):
        """Test that invalid bin size raises ValueError."""
        spikes = np.array([0.1, 0.2])
        
        # Test negative bin size
        with pytest.raises(ValueError, match="Bin size must be positive"):
            bin_spike_train(spikes, bin_size=-0.1, duration=1.0)
        
        # Test zero bin size
        with pytest.raises(ValueError, match="Bin size must be positive"):
            bin_spike_train(spikes, bin_size=0.0, duration=1.0)
    
    def test_empty_bins(self):
        """Test that bins without spikes are zero."""
        spikes = np.array([0.5])
        counts = bin_spike_train(spikes, bin_size=0.1, duration=1.0)
        
        # Most bins should be zero
        assert np.sum(counts == 0) == 9
        # One bin should have 1 spike
        assert np.sum(counts == 1) == 1
        assert counts[5] == 1  # Spike at 0.5 should be in bin 5


class TestBurstDetection:
    """Tests for burst detection."""
    
    def test_detect_bursts(self, burst_spikes):
        """Test burst detection with clear bursts."""
        bursts = detect_bursts(burst_spikes, max_isi=0.01, min_spikes=3)
        
        # Should detect 2 bursts
        assert len(bursts) == 2
        
        # First burst: indices 0-3
        assert bursts[0] == (0, 3)
        
        # Second burst: indices 4-7
        assert bursts[1] == (4, 7)
    
    def test_no_bursts_in_regular(self, regular_spikes):
        """Regular spiking should have no bursts."""
        # With tight ISI threshold, regular 100ms ISI won't form bursts
        bursts = detect_bursts(regular_spikes, max_isi=0.01, min_spikes=3)
        assert len(bursts) == 0
    
    def test_min_spikes_threshold(self):
        """Test minimum spike requirement for burst."""
        # Burst with only 2 spikes
        spikes = np.array([0.1, 0.105, 0.5])
        bursts = detect_bursts(spikes, max_isi=0.01, min_spikes=3)
        
        # Should not detect burst (need min 3 spikes)
        assert len(bursts) == 0
    
    def test_single_long_burst(self):
        """Test detection of single long burst."""
        # Create burst with 5 spikes
        spikes = np.array([0.1, 0.105, 0.11, 0.115, 0.12])
        bursts = detect_bursts(spikes, max_isi=0.01, min_spikes=3)
        
        # Should detect 1 burst spanning all spikes
        assert len(bursts) == 1
        assert bursts[0] == (0, 4)
    
    def test_no_bursts_insufficient_spikes(self):
        """Test with fewer spikes than min_spikes."""
        spikes = np.array([0.1, 0.105])
        bursts = detect_bursts(spikes, max_isi=0.01, min_spikes=3)
        
        # Should return empty list
        assert len(bursts) == 0


# Additional test cases
class TestEdgeCases:
    """Additional edge case tests."""
    
    def test_firing_rate_with_large_duration(self):
        """Test firing rate with large duration."""
        spikes = np.array([0.1, 0.2, 0.3])
        rate = calculate_firing_rate(spikes, duration=100.0)
        assert rate == pytest.approx(0.03)
    
    def test_cv_with_identical_isis(self):
        """Test CV with perfectly regular spikes."""
        spikes = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        cv = calculate_cv(spikes)
        assert cv == pytest.approx(0.0, abs=1e-10)
    
    def test_binning_with_multiple_spikes_per_bin(self):
        """Test binning when multiple spikes fall in same bin."""
        spikes = np.array([0.01, 0.02, 0.03, 0.11, 0.12])
        counts = bin_spike_train(spikes, bin_size=0.1, duration=0.5)
        
        # First bin should have 3 spikes
        assert counts[0] == 3
        # Second bin should have 2 spikes
        assert counts[1] == 2
    
    def test_burst_detection_at_end(self):
        """Test burst detection when burst is at end of spike train."""
        spikes = np.array([0.1, 0.5, 0.9, 0.905, 0.91, 0.915])
        bursts = detect_bursts(spikes, max_isi=0.01, min_spikes=3)
        
        # Should detect burst at end
        assert len(bursts) == 1
        assert bursts[0][1] == 5  # Last index


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
