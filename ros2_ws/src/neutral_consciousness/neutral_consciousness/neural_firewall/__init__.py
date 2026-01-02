"""
Neural Firewall Subpackage

Security middleware for monitoring neural traffic and preventing
unauthorized access or "brainjacking" during consciousness operations.

Modules:
- traffic_monitor: Spike rate monitoring and kill switch
- homomorphic_encryption: Encrypted neural processing for satellite link
"""

from .traffic_monitor import NeuralFirewall
from .homomorphic_encryption import (
    HomomorphicContext,
    NeuralEncryptionWrapper,
    EncryptedVector
)
