"""
Bonus Challenge: Complete Pipeline
=====================================================

Neural Spike Analysis Pipeline demonstrating all Day 7 concepts:
- OOP with dataclasses and protocols
- Error handling
- Type hints
- Functional programming
- Testing
- Context managers
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Protocol
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
import json


@dataclass
class SpikeTrainStats:
    """Statistics for a single spike train."""
    neuron_id: str
    n_spikes: int
    firing_rate: float
    cv_isi: float
    mean_isi: float
    burst_count: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class SpikeAnalyzer(Protocol):
    """Protocol for spike analyzers."""
    def analyze(self, spike_times: NDArray) -> dict:
        """Analyze spike train and return statistics."""
        ...


class BasicSpikeAnalyzer:
    """Basic implementation of spike analyzer."""
    
    def __init__(self, duration: float):
        """Initialize analyzer.
        
        Parameters
        ----------
        duration : float
            Recording duration in seconds
        """
        self.duration = duration
    
    def analyze(self, spike_times: NDArray) -> dict:
        """Analyze spike train and return statistics.
        
        Parameters
        ----------
        spike_times : NDArray
            Array of spike times in seconds
            
        Returns
        -------
        dict
            Dictionary with analysis results
        """
        n_spikes = len(spike_times)
        
        # Calculate firing rate
        firing_rate = n_spikes / self.duration if self.duration > 0 else 0.0
        
        # Calculate ISI statistics
        if n_spikes >= 2:
            isis = np.diff(spike_times)
            mean_isi = float(np.mean(isis))
            cv_isi = float(np.std(isis) / mean_isi) if mean_isi > 0 else np.inf
        else:
            mean_isi = 0.0
            cv_isi = 0.0
        
        # Detect bursts (simple implementation)
        burst_count = self._detect_bursts(spike_times)
        
        return {
            'n_spikes': n_spikes,
            'firing_rate': firing_rate,
            'cv_isi': cv_isi,
            'mean_isi': mean_isi,
            'burst_count': burst_count
        }
    
    def _detect_bursts(self, spike_times: NDArray, 
                      max_isi: float = 0.01, 
                      min_spikes: int = 3) -> int:
        """Detect number of bursts in spike train."""
        if len(spike_times) < min_spikes:
            return 0
        
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
        
        return len(bursts)


class NeuralDataset:
    """Container for multiple neurons' spike data."""
    
    def __init__(self):
        """Initialize empty dataset."""
        self.neurons: Dict[str, NDArray] = {}
        self.duration: float = 0.0
        self.metadata: dict = {}
    
    @classmethod
    def from_directory(cls, data_dir: Path) -> 'NeuralDataset':
        """Load all spike data from directory.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing spike data files (CSV or NPY)
            
        Returns
        -------
        NeuralDataset
            Dataset with loaded neurons
            
        Raises
        ------
        FileNotFoundError
            If directory doesn't exist
        ValueError
            If no valid data files found
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        dataset = cls()
        
        # Load all .npy files
        npy_files = list(data_dir.glob('*.npy'))
        
        if not npy_files:
            raise ValueError(f"No .npy files found in {data_dir}")
        
        for file_path in npy_files:
            neuron_id = file_path.stem
            spike_times = np.load(file_path)
            dataset.add_neuron(neuron_id, spike_times)
        
        return dataset
    
    def add_neuron(self, neuron_id: str, spike_times: NDArray) -> None:
        """Add neuron to dataset.
        
        Parameters
        ----------
        neuron_id : str
            Unique identifier for neuron
        spike_times : NDArray
            Array of spike times in seconds
            
        Raises
        ------
        ValueError
            If neuron_id already exists or spike_times is invalid
        """
        if neuron_id in self.neurons:
            raise ValueError(f"Neuron {neuron_id} already exists in dataset")
        
        if not isinstance(spike_times, np.ndarray):
            raise ValueError("spike_times must be a numpy array")
        
        if len(spike_times) > 0:
            # Validate spike times are sorted and non-negative
            if not np.all(spike_times >= 0):
                raise ValueError("Spike times must be non-negative")
            
            if not np.all(np.diff(spike_times) >= 0):
                # Sort if not sorted
                spike_times = np.sort(spike_times)
        
        self.neurons[neuron_id] = spike_times
        
        # Update duration
        if len(spike_times) > 0:
            max_time = float(np.max(spike_times))
            self.duration = max(self.duration, max_time)
    
    def get_neuron(self, neuron_id: str) -> Optional[NDArray]:
        """Get spike times for neuron.
        
        Parameters
        ----------
        neuron_id : str
            Neuron identifier
            
        Returns
        -------
        Optional[NDArray]
            Spike times array, or None if not found
        """
        return self.neurons.get(neuron_id)
    
    def filter_by_rate(self, min_rate: float, max_rate: float) -> 'NeuralDataset':
        """Filter neurons by firing rate.
        
        Parameters
        ----------
        min_rate : float
            Minimum firing rate in Hz
        max_rate : float
            Maximum firing rate in Hz
            
        Returns
        -------
        NeuralDataset
            New dataset with filtered neurons
        """
        if self.duration == 0:
            raise ValueError("Cannot filter by rate: duration is 0")
        
        # Use functional programming: filter
        filtered_neurons = {
            neuron_id: spikes
            for neuron_id, spikes in self.neurons.items()
            if min_rate <= (len(spikes) / self.duration) <= max_rate
        }
        
        # Create new dataset
        new_dataset = NeuralDataset()
        new_dataset.duration = self.duration
        new_dataset.neurons = filtered_neurons
        new_dataset.metadata = self.metadata.copy()
        
        return new_dataset
    
    def analyze_all(self, analyzer: Optional[SpikeAnalyzer] = None) -> List[SpikeTrainStats]:
        """Analyze all neurons in dataset.
        
        Parameters
        ----------
        analyzer : Optional[SpikeAnalyzer]
            Analyzer to use. If None, uses BasicSpikeAnalyzer
            
        Returns
        -------
        List[SpikeTrainStats]
            List of statistics for all neurons
        """
        if analyzer is None:
            analyzer = BasicSpikeAnalyzer(self.duration)
        
        # Use functional programming: map
        results = []
        for neuron_id, spike_times in self.neurons.items():
            stats_dict = analyzer.analyze(spike_times)
            stats = SpikeTrainStats(
                neuron_id=neuron_id,
                **stats_dict
            )
            results.append(stats)
        
        return results
    
    def to_json(self, filename: Path) -> None:
        """Export dataset to JSON.
        
        Parameters
        ----------
        filename : Path
            Output JSON file path
        """
        # Convert numpy arrays to lists for JSON serialization
        data = {
            'neurons': {
                neuron_id: spikes.tolist()
                for neuron_id, spikes in self.neurons.items()
            },
            'duration': self.duration,
            'metadata': self.metadata
        }
        
        # Use context manager
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_json(cls, filename: Path) -> 'NeuralDataset':
        """Load dataset from JSON.
        
        Parameters
        ----------
        filename : Path
            Input JSON file path
            
        Returns
        -------
        NeuralDataset
            Loaded dataset
            
        Raises
        ------
        FileNotFoundError
            If file doesn't exist
        ValueError
            If JSON is invalid
        """
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")
        
        # Use context manager
        with open(filename, 'r') as f:
            data = json.load(f)
        
        dataset = cls()
        dataset.duration = data.get('duration', 0.0)
        dataset.metadata = data.get('metadata', {})
        
        # Convert lists back to numpy arrays
        for neuron_id, spikes_list in data.get('neurons', {}).items():
            dataset.neurons[neuron_id] = np.array(spikes_list)
        
        return dataset
    
    def __len__(self) -> int:
        """Return number of neurons in dataset."""
        return len(self.neurons)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"NeuralDataset(n_neurons={len(self)}, duration={self.duration:.2f}s)"


class AnalysisPipeline:
    """Complete analysis pipeline."""
    
    def __init__(self, dataset: NeuralDataset):
        """Initialize pipeline.
        
        Parameters
        ----------
        dataset : NeuralDataset
            Dataset to analyze
        """
        self.dataset = dataset
        self.results: Optional[List[SpikeTrainStats]] = None
    
    def run(self, analyzer: Optional[SpikeAnalyzer] = None) -> List[SpikeTrainStats]:
        """Run complete analysis pipeline.
        
        Parameters
        ----------
        analyzer : Optional[SpikeAnalyzer]
            Custom analyzer to use
            
        Returns
        -------
        List[SpikeTrainStats]
            Analysis results for all neurons
            
        Raises
        ------
        ValueError
            If dataset is empty
        """
        # 1. Validate data
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # 2. Analyze each neuron
        self.results = self.dataset.analyze_all(analyzer)
        
        # 3. Return summary
        return self.results
    
    def generate_report(self, output_file: Path) -> None:
        """Generate analysis report.
        
        Parameters
        ----------
        output_file : Path
            Output markdown file path
            
        Raises
        ------
        ValueError
            If pipeline hasn't been run yet
        """
        if self.results is None:
            raise ValueError("Pipeline must be run before generating report")
        
        # Use context manager
        with open(output_file, 'w') as f:
            f.write("# Neural Spike Analysis Report\n\n")
            f.write(f"**Dataset:** {len(self.dataset)} neurons\n")
            f.write(f"**Duration:** {self.dataset.duration:.2f} seconds\n\n")
            
            f.write("## Summary Statistics\n\n")
            
            # Calculate summary stats
            firing_rates = [r.firing_rate for r in self.results]
            cvs = [r.cv_isi for r in self.results if r.n_spikes >= 2]
            
            f.write(f"- **Mean firing rate:** {np.mean(firing_rates):.2f} Hz\n")
            f.write(f"- **Median firing rate:** {np.median(firing_rates):.2f} Hz\n")
            if cvs:
                f.write(f"- **Mean CV:** {np.mean(cvs):.3f}\n")
            
            f.write("\n## Individual Neurons\n\n")
            f.write("| Neuron ID | Spikes | Rate (Hz) | CV | Mean ISI (ms) | Bursts |\n")
            f.write("|-----------|--------|-----------|----|--------------:|--------|\n")
            
            for stats in sorted(self.results, key=lambda x: x.firing_rate, reverse=True):
                f.write(
                    f"| {stats.neuron_id} | {stats.n_spikes} | "
                    f"{stats.firing_rate:.2f} | {stats.cv_isi:.3f} | "
                    f"{stats.mean_isi*1000:.2f} | {stats.burst_count} |\n"
                )


# ============================================================================
# TESTS
# ============================================================================

def test_spike_train_stats():
    """Test SpikeTrainStats dataclass."""
    print("Testing SpikeTrainStats...")
    
    stats = SpikeTrainStats(
        neuron_id='n1',
        n_spikes=100,
        firing_rate=10.0,
        cv_isi=0.5,
        mean_isi=0.1,
        burst_count=5
    )
    
    assert stats.neuron_id == 'n1'
    assert stats.n_spikes == 100
    
    # Test to_dict
    d = stats.to_dict()
    assert d['neuron_id'] == 'n1'
    assert d['firing_rate'] == 10.0
    
    print("✓ SpikeTrainStats tests passed\n")


def test_neural_dataset():
    """Test NeuralDataset class."""
    print("Testing NeuralDataset...")
    
    # Create dataset
    dataset = NeuralDataset()
    assert len(dataset) == 0
    
    # Add neurons
    spikes1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    spikes2 = np.array([0.15, 0.25, 0.35, 0.45])
    
    dataset.add_neuron('n1', spikes1)
    dataset.add_neuron('n2', spikes2)
    
    assert len(dataset) == 2
    assert dataset.duration == 0.5
    print("  ✓ Adding neurons works")
    
    # Test get_neuron
    retrieved = dataset.get_neuron('n1')
    assert np.array_equal(retrieved, spikes1)
    print("  ✓ Getting neurons works")
    
    # Test filter_by_rate
    dataset.duration = 1.0  # Set explicit duration
    filtered = dataset.filter_by_rate(4.0, 6.0)
    assert len(filtered) == 2  # Both neurons have 4-5 spikes
    print("  ✓ Filtering by rate works")
    
    # Test analyze_all
    results = dataset.analyze_all()
    assert len(results) == 2
    assert all(isinstance(r, SpikeTrainStats) for r in results)
    print("  ✓ Analyzing all neurons works")
    
    print("✓ NeuralDataset tests passed\n")


def test_json_serialization():
    """Test JSON save/load."""
    print("Testing JSON serialization...")
    
    # Create dataset
    dataset = NeuralDataset()
    dataset.add_neuron('n1', np.array([0.1, 0.2, 0.3]))
    dataset.add_neuron('n2', np.array([0.15, 0.25]))
    dataset.duration = 1.0
    dataset.metadata = {'experiment': 'test'}
    
    # Save to JSON
    json_file = Path('test_dataset.json')
    dataset.to_json(json_file)
    print("  ✓ Saving to JSON works")
    
    # Load from JSON
    loaded = NeuralDataset.from_json(json_file)
    assert len(loaded) == 2
    assert loaded.duration == 1.0
    assert loaded.metadata['experiment'] == 'test'
    assert np.array_equal(loaded.get_neuron('n1'), np.array([0.1, 0.2, 0.3]))
    print("  ✓ Loading from JSON works")
    
    # Cleanup
    json_file.unlink()
    
    print("✓ JSON serialization tests passed\n")


def test_analysis_pipeline():
    """Test AnalysisPipeline."""
    print("Testing AnalysisPipeline...")
    
    # Create dataset
    dataset = NeuralDataset()
    dataset.add_neuron('n1', np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    dataset.add_neuron('n2', np.array([0.15, 0.25, 0.35, 0.45]))
    dataset.duration = 1.0
    
    # Create pipeline
    pipeline = AnalysisPipeline(dataset)
    
    # Run analysis
    results = pipeline.run()
    assert len(results) == 2
    assert all(isinstance(r, SpikeTrainStats) for r in results)
    print("  ✓ Running pipeline works")
    
    # Generate report
    report_file = Path('test_report.md')
    pipeline.generate_report(report_file)
    assert report_file.exists()
    print("  ✓ Generating report works")
    
    # Check report content
    with open(report_file, 'r') as f:
        content = f.read()
        assert 'Neural Spike Analysis Report' in content
        assert 'n1' in content
        assert 'n2' in content
    
    # Cleanup
    report_file.unlink()
    
    print("✓ AnalysisPipeline tests passed\n")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING ALL TESTS FOR COMPLETE PIPELINE")
    print("="*60 + "\n")
    
    test_spike_train_stats()
    test_neural_dataset()
    test_json_serialization()
    test_analysis_pipeline()
    
    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)


if __name__ == '__main__':
    run_all_tests()
