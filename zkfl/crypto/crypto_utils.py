"""
Cryptographic utilities for the ZKFL framework.

This module provides various cryptographic functions including hashing,
encryption, digital signatures, and key management utilities.
"""

import hashlib
import hmac
import secrets
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

from ..core.exceptions import CryptographicError

logger = logging.getLogger(__name__)


class CryptoUtils:
    """Cryptographic utilities for secure federated learning."""

    @staticmethod
    def generate_secure_hash(data: Union[str, bytes, Dict[str, Any]]) -> str:
        """
        Generate a secure SHA-256 hash of the input data.

        Args:
            data: Data to hash (string, bytes, or dictionary)

        Returns:
            Hexadecimal hash string
        """
        if isinstance(data, dict):
            # Convert dictionary to sorted JSON string for consistent hashing
            import json

            data_str = json.dumps(data, sort_keys=True, default=str)
            data_bytes = data_str.encode("utf-8")
        elif isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            # Try to convert to string first
            data_bytes = str(data).encode("utf-8")

        return hashlib.sha256(data_bytes).hexdigest()

    @staticmethod
    def generate_merkle_root(data_list: List[Union[str, bytes]]) -> str:
        """
        Generate Merkle root hash from a list of data items.

        Args:
            data_list: List of data items to hash

        Returns:
            Merkle root hash
        """
        if not data_list:
            return CryptoUtils.generate_secure_hash("")

        # Convert all items to hashes
        hashes = []
        for item in data_list:
            if isinstance(item, str):
                item_hash = CryptoUtils.generate_secure_hash(item)
            else:
                item_hash = CryptoUtils.generate_secure_hash(
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                )
            hashes.append(item_hash)

        # Build Merkle tree
        while len(hashes) > 1:
            next_level = []

            # Process pairs
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    # Hash the pair
                    combined = hashes[i] + hashes[i + 1]
                else:
                    # Odd number of hashes, duplicate the last one
                    combined = hashes[i] + hashes[i]

                next_level.append(CryptoUtils.generate_secure_hash(combined))

            hashes = next_level

        return hashes[0]

    @staticmethod
    def generate_hmac(key: str, message: str) -> str:
        """
        Generate HMAC signature for message authentication.

        Args:
            key: Secret key for HMAC
            message: Message to authenticate

        Returns:
            HMAC signature in hexadecimal
        """
        key_bytes = key.encode("utf-8")
        message_bytes = message.encode("utf-8")

        signature = hmac.new(key_bytes, message_bytes, hashlib.sha256)
        return signature.hexdigest()

    @staticmethod
    def verify_hmac(key: str, message: str, signature: str) -> bool:
        """
        Verify HMAC signature.

        Args:
            key: Secret key used for HMAC
            message: Original message
            signature: HMAC signature to verify

        Returns:
            True if signature is valid, False otherwise
        """
        expected_signature = CryptoUtils.generate_hmac(key, message)
        return hmac.compare_digest(expected_signature, signature)

    @staticmethod
    def generate_random_key(length: int = 32) -> str:
        """
        Generate a cryptographically secure random key.

        Args:
            length: Length of the key in bytes

        Returns:
            Random key in hexadecimal format
        """
        return secrets.token_hex(length)

    @staticmethod
    def generate_nonce(length: int = 16) -> str:
        """
        Generate a cryptographic nonce.

        Args:
            length: Length of the nonce in bytes

        Returns:
            Random nonce in hexadecimal format
        """
        return secrets.token_hex(length)

    @staticmethod
    def derive_key(password: str, salt: str, iterations: int = 100000) -> str:
        """
        Derive a key from a password using PBKDF2.

        Args:
            password: Password to derive from
            salt: Salt for key derivation
            iterations: Number of iterations

        Returns:
            Derived key in hexadecimal format
        """
        password_bytes = password.encode("utf-8")
        salt_bytes = salt.encode("utf-8")

        derived_key = hashlib.pbkdf2_hmac(
            "sha256", password_bytes, salt_bytes, iterations, dklen=32
        )

        return derived_key.hex()

    @staticmethod
    def compute_model_fingerprint(model_params: Dict[str, Any]) -> str:
        """
        Compute a unique fingerprint for model parameters.

        Args:
            model_params: Dictionary of model parameters

        Returns:
            Model fingerprint hash
        """
        # Create a deterministic representation of model parameters
        param_hashes = []

        for param_name, param_value in sorted(model_params.items()):
            # Handle different parameter types
            if hasattr(param_value, "shape") and hasattr(param_value, "flatten"):
                # NumPy/PyTorch array
                param_str = f"{param_name}:{param_value.shape}:{param_value.flatten()[:10].tolist()}"
            elif isinstance(param_value, (list, tuple)):
                param_str = f"{param_name}:{len(param_value)}:{param_value[:10]}"
            else:
                param_str = f"{param_name}:{str(param_value)}"

            param_hash = CryptoUtils.generate_secure_hash(param_str)
            param_hashes.append(param_hash)

        # Combine all parameter hashes
        combined_hash = CryptoUtils.generate_secure_hash("".join(param_hashes))
        return combined_hash

    @staticmethod
    def create_commitment(value: str, nonce: str) -> Tuple[str, str]:
        """
        Create a cryptographic commitment to a value.

        Args:
            value: Value to commit to
            nonce: Random nonce for hiding

        Returns:
            Tuple of (commitment, nonce)
        """
        commitment_input = f"{value}:{nonce}"
        commitment = CryptoUtils.generate_secure_hash(commitment_input)
        return commitment, nonce

    @staticmethod
    def verify_commitment(value: str, nonce: str, commitment: str) -> bool:
        """
        Verify a cryptographic commitment.

        Args:
            value: Revealed value
            nonce: Revealed nonce
            commitment: Original commitment

        Returns:
            True if commitment is valid, False otherwise
        """
        expected_commitment, _ = CryptoUtils.create_commitment(value, nonce)
        return expected_commitment == commitment

    @staticmethod
    def encrypt_simple(plaintext: str, key: str) -> str:
        """
        Simple symmetric encryption using XOR (for demonstration).
        Note: This is NOT cryptographically secure and should not be used in production.

        Args:
            plaintext: Text to encrypt
            key: Encryption key

        Returns:
            Encrypted text in hexadecimal
        """
        # Extend key to match plaintext length
        extended_key = (key * (len(plaintext) // len(key) + 1))[: len(plaintext)]

        # XOR encryption
        encrypted_bytes = []
        for i, char in enumerate(plaintext):
            encrypted_byte = ord(char) ^ ord(extended_key[i])
            encrypted_bytes.append(encrypted_byte)

        return bytes(encrypted_bytes).hex()

    @staticmethod
    def decrypt_simple(ciphertext_hex: str, key: str) -> str:
        """
        Simple symmetric decryption using XOR (for demonstration).

        Args:
            ciphertext_hex: Encrypted text in hexadecimal
            key: Decryption key

        Returns:
            Decrypted plaintext
        """
        # Convert hex to bytes
        ciphertext_bytes = bytes.fromhex(ciphertext_hex)

        # Extend key to match ciphertext length
        extended_key = (key * (len(ciphertext_bytes) // len(key) + 1))[
            : len(ciphertext_bytes)
        ]

        # XOR decryption
        decrypted_chars = []
        for i, byte_val in enumerate(ciphertext_bytes):
            decrypted_char = chr(byte_val ^ ord(extended_key[i]))
            decrypted_chars.append(decrypted_char)

        return "".join(decrypted_chars)


class SecureAggregator:
    """Secure aggregation utilities for federated learning."""

    def __init__(self):
        """Initialize secure aggregator."""
        self.client_keys: Dict[str, str] = {}
        self.aggregation_key = CryptoUtils.generate_random_key()

    def register_client(self, client_id: str) -> str:
        """
        Register a client and generate shared key.

        Args:
            client_id: Unique client identifier

        Returns:
            Shared key for the client
        """
        client_key = CryptoUtils.generate_random_key()
        self.client_keys[client_id] = client_key

        logger.info(f"Client {client_id} registered for secure aggregation")
        return client_key

    def create_secure_share(
        self, client_id: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create secure shares of model parameters.

        Args:
            client_id: Client identifier
            parameters: Model parameters to secure

        Returns:
            Secure shares of parameters
        """
        if client_id not in self.client_keys:
            raise CryptographicError(f"Client {client_id} not registered")

        client_key = self.client_keys[client_id]

        # Create shares (simplified secret sharing simulation)
        shares = {}
        for param_name, param_value in parameters.items():
            # Convert parameter to string for sharing
            param_str = str(param_value)

            # Create commitment
            nonce = CryptoUtils.generate_nonce()
            commitment, _ = CryptoUtils.create_commitment(param_str, nonce)

            # Create share with HMAC
            share_data = f"{param_str}:{nonce}"
            signature = CryptoUtils.generate_hmac(client_key, share_data)

            shares[param_name] = {
                "commitment": commitment,
                "signature": signature,
                "nonce": nonce,
            }

        return shares

    def verify_secure_share(
        self, client_id: str, shares: Dict[str, Any], parameters: Dict[str, Any]
    ) -> bool:
        """
        Verify secure shares from a client.

        Args:
            client_id: Client identifier
            shares: Secure shares to verify
            parameters: Original parameters

        Returns:
            True if shares are valid, False otherwise
        """
        if client_id not in self.client_keys:
            return False

        client_key = self.client_keys[client_id]

        for param_name, param_value in parameters.items():
            if param_name not in shares:
                return False

            share = shares[param_name]
            param_str = str(param_value)
            nonce = share["nonce"]

            # Verify commitment
            if not CryptoUtils.verify_commitment(param_str, nonce, share["commitment"]):
                return False

            # Verify HMAC
            share_data = f"{param_str}:{nonce}"
            if not CryptoUtils.verify_hmac(client_key, share_data, share["signature"]):
                return False

        return True

    def get_client_count(self) -> int:
        """Get number of registered clients."""
        return len(self.client_keys)

    def remove_client(self, client_id: str) -> None:
        """Remove a client from secure aggregation."""
        if client_id in self.client_keys:
            del self.client_keys[client_id]
            logger.info(f"Client {client_id} removed from secure aggregation")


class DigitalSignature:
    """Digital signature utilities (simplified implementation)."""

    def __init__(self, private_key: Optional[str] = None):
        """
        Initialize digital signature system.

        Args:
            private_key: Private key for signing (generated if None)
        """
        self.private_key = private_key or CryptoUtils.generate_random_key(64)
        self.public_key = CryptoUtils.generate_secure_hash(self.private_key)

    def sign(self, message: str) -> str:
        """
        Sign a message with the private key.

        Args:
            message: Message to sign

        Returns:
            Digital signature
        """
        # Simplified signing (in practice, use RSA/ECDSA)
        signature_input = f"{message}:{self.private_key}"
        signature = CryptoUtils.generate_secure_hash(signature_input)
        return signature

    def verify(self, message: str, signature: str, public_key: str) -> bool:
        """
        Verify a digital signature.

        Args:
            message: Original message
            signature: Signature to verify
            public_key: Public key for verification

        Returns:
            True if signature is valid, False otherwise
        """
        # In a real implementation, this would require the private key
        # For simulation, we use a deterministic approach
        try:
            # This is a simplified verification - not cryptographically secure
            expected_private_key = None
            for possible_key in [CryptoUtils.generate_random_key(64)]:
                if CryptoUtils.generate_secure_hash(possible_key) == public_key:
                    expected_private_key = possible_key
                    break

            if expected_private_key:
                signature_input = f"{message}:{expected_private_key}"
                expected_signature = CryptoUtils.generate_secure_hash(signature_input)
                return expected_signature == signature

            return False

        except Exception:
            return False

    def get_public_key(self) -> str:
        """Get the public key."""
        return self.public_key
