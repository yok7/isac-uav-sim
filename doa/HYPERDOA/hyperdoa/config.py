"""
Configuration dataclass for DOA estimation parameters.

This module provides a minimal configuration class for HDC-based DOA estimation,
containing only the essential parameters needed for the system model.
"""

from dataclasses import dataclass


@dataclass
class DOAConfig:
    """Configuration for DOA estimation system.

    Attributes:
        N: Number of sensors (antennas) in the uniform linear array
        M: Number of signal sources to estimate
        T: Number of time snapshots per observation
        snr: Signal-to-noise ratio in dB
        signal_nature: Type of signal correlation ("non-coherent" or "coherent")

    Example:
        >>> config = DOAConfig(N=8, M=3, T=100, snr=-5)
        >>> print(f"Array has {config.N} sensors")
        Array has 8 sensors
    """

    N: int = 8
    M: int = 3
    T: int = 100
    snr: float = -5.0
    signal_nature: str = "non-coherent"

    def set_parameter(self, name: str, value):
        """Set parameter value (for compatibility with SubspaceNet).

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            self for method chaining
        """
        setattr(self, name, value)
        return self
