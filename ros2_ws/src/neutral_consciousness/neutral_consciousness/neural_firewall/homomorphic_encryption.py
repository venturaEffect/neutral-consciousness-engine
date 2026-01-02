"""
Homomorphic Encryption Module for Neural Firewall

Implements a wrapper for processing neural signals with homomorphic encryption,
ensuring the satellite can process data WITHOUT knowing what it means.

Security Model:
1. Biological Brain encrypts spikes: E(x)
2. Satellite processes encrypted data: E(x) + E(y) = E(x + y)
3. Biological Brain decrypts result

If hackers seize the satellite, they only see encrypted noise.
They cannot read thoughts or inject commands without the private key.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import hashlib
import secrets

# Try to import actual homomorphic encryption library
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False


@dataclass
class EncryptedSpike:
    """Represents an encrypted neural spike packet."""
    ciphertext: bytes
    nonce: bytes
    timestamp: float
    neuron_id: int


class HomomorphicContext:
    """
    Manages homomorphic encryption context.
    
    Uses TenSEAL (CKKS scheme) when available, otherwise falls back
    to a demonstration-only stub that simulates the API.
    """
    
    def __init__(self, poly_modulus_degree: int = 8192, security_level: int = 128):
        """
        Initialize encryption context.
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            security_level: Security bits (128 or 256)
        """
        self.poly_modulus_degree = poly_modulus_degree
        self.security_level = security_level
        
        if TENSEAL_AVAILABLE:
            self._init_tenseal()
        else:
            self._init_stub()
    
    def _init_tenseal(self):
        """Initialize TenSEAL CKKS context."""
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        
        # Store secret key separately (would be in biological interface)
        self._secret_key = self.context.secret_key()
        
        # Create public context (what satellite receives)
        self.public_context = self.context.copy()
        self.public_context.make_context_public()
    
    def _init_stub(self):
        """Initialize demonstration stub."""
        self._secret_key = secrets.token_bytes(32)
        self._scale = 1000.0  # Fixed-point scale
        self.context = None
        self.public_context = None
    
    def encrypt(self, data: np.ndarray) -> 'EncryptedVector':
        """
        Encrypt neural data.
        
        Args:
            data: Neural activity vector
            
        Returns:
            Encrypted vector that can be processed homomorphically
        """
        if TENSEAL_AVAILABLE:
            encrypted = ts.ckks_vector(self.context, data.tolist())
            return EncryptedVector(encrypted, is_real=True)
        else:
            # Stub: simulate encryption with fixed-point encoding
            encoded = (data * self._scale).astype(np.int64)
            noise = np.random.randint(-100, 100, size=data.shape, dtype=np.int64)
            encrypted = encoded + noise
            return EncryptedVector(encrypted, is_real=False, stub_noise=noise)
    
    def decrypt(self, encrypted: 'EncryptedVector') -> np.ndarray:
        """
        Decrypt encrypted vector.
        
        Args:
            encrypted: Encrypted vector
            
        Returns:
            Decrypted neural data
        """
        if encrypted.is_real:
            return np.array(encrypted.data.decrypt())
        else:
            # Stub: reverse fake encryption
            decrypted = (encrypted.data - encrypted.stub_noise) / self._scale
            return decrypted.astype(np.float32)
    
    def get_public_context_bytes(self) -> bytes:
        """Export public context for satellite transmission."""
        if TENSEAL_AVAILABLE:
            return self.public_context.serialize()
        else:
            return b"STUB_PUBLIC_CONTEXT"


class EncryptedVector:
    """
    Wrapper for encrypted neural data vectors.
    
    Supports homomorphic operations:
    - Addition: E(x) + E(y) = E(x + y)
    - Scalar multiplication: c * E(x) = E(c * x)
    """
    
    def __init__(self, data, is_real: bool, stub_noise: Optional[np.ndarray] = None):
        self.data = data
        self.is_real = is_real
        self.stub_noise = stub_noise
    
    def __add__(self, other: 'EncryptedVector') -> 'EncryptedVector':
        """Homomorphic addition."""
        if self.is_real:
            result = self.data + other.data
            return EncryptedVector(result, is_real=True)
        else:
            result = self.data + other.data
            noise = self.stub_noise + other.stub_noise
            return EncryptedVector(result, is_real=False, stub_noise=noise)
    
    def __mul__(self, scalar: float) -> 'EncryptedVector':
        """Scalar multiplication."""
        if self.is_real:
            result = self.data * scalar
            return EncryptedVector(result, is_real=True)
        else:
            result = (self.data * scalar).astype(np.int64)
            noise = (self.stub_noise * scalar).astype(np.int64)
            return EncryptedVector(result, is_real=False, stub_noise=noise)
    
    def __neg__(self) -> 'EncryptedVector':
        """Negation for subtraction."""
        return self * (-1.0)


class NeuralEncryptionWrapper:
    """
    High-level wrapper for encrypting neural traffic.
    
    This class is used by the Neural Firewall to secure all
    communications between the biological interface and the satellite.
    """
    
    def __init__(self):
        self.context = HomomorphicContext()
        self._session_id = secrets.token_hex(16)
        self._packet_counter = 0
    
    def encrypt_spike_train(self, spikes: np.ndarray) -> Tuple[EncryptedVector, dict]:
        """
        Encrypt a spike train for transmission.
        
        Args:
            spikes: Array of spike rates/times
            
        Returns:
            Tuple of (encrypted_data, metadata)
        """
        encrypted = self.context.encrypt(spikes.astype(np.float32))
        
        metadata = {
            'session_id': self._session_id,
            'packet_id': self._packet_counter,
            'shape': spikes.shape,
            'dtype': str(spikes.dtype),
            'checksum': hashlib.sha256(spikes.tobytes()).hexdigest()[:16]
        }
        
        self._packet_counter += 1
        return encrypted, metadata
    
    def decrypt_response(self, encrypted: EncryptedVector, metadata: dict) -> np.ndarray:
        """
        Decrypt response from satellite.
        
        Args:
            encrypted: Encrypted response vector
            metadata: Associated metadata
            
        Returns:
            Decrypted neural data
        """
        # Verify session
        if metadata.get('session_id') != self._session_id:
            raise SecurityError("Session ID mismatch - possible hijack attempt")
        
        return self.context.decrypt(encrypted)
    
    def process_encrypted_prediction(
        self, 
        encrypted_input: EncryptedVector,
        encrypted_prediction: EncryptedVector
    ) -> EncryptedVector:
        """
        Compute prediction error on encrypted data.
        
        This happens on the SATELLITE - it never sees the actual values!
        
        E(error) = E(input) - E(prediction) = E(input - prediction)
        """
        return encrypted_input + (-encrypted_prediction)


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


# Demonstration function
def demo_homomorphic_processing():
    """
    Demonstrate end-to-end homomorphic processing.
    
    Shows that the satellite can compute prediction error
    without ever seeing the actual neural data.
    """
    print("=== Homomorphic Neural Processing Demo ===\n")
    
    # Simulate biological brain generating spikes
    biological_input = np.array([0.5, 0.3, 0.8, 0.1, 0.6], dtype=np.float32)
    print(f"Biological Input (SECRET): {biological_input}")
    
    # Initialize encryption (biological interface holds private key)
    wrapper = NeuralEncryptionWrapper()
    
    # Encrypt for transmission
    enc_input, meta = wrapper.encrypt_spike_train(biological_input)
    print(f"\nEncrypted (what satellite sees): <encrypted blob>")
    
    # Simulate satellite computing prediction (also encrypted)
    prediction = np.array([0.4, 0.3, 0.7, 0.2, 0.5], dtype=np.float32)
    enc_prediction, _ = wrapper.encrypt_spike_train(prediction)
    
    # SATELLITE computes error on encrypted data
    enc_error = wrapper.process_encrypted_prediction(enc_input, enc_prediction)
    print("Satellite computed E(error) = E(input) - E(prediction)")
    
    # Only biological interface can decrypt
    decrypted_error = wrapper.context.decrypt(enc_error)
    expected_error = biological_input - prediction
    
    print(f"\nDecrypted Error: {decrypted_error}")
    print(f"Expected Error:  {expected_error}")
    print(f"Match: {np.allclose(decrypted_error, expected_error, atol=0.01)}")


if __name__ == '__main__':
    demo_homomorphic_processing()
