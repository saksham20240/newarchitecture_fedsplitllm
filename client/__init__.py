"""
Federated learning client package
"""

from .federated_client import FederatedMedicalClient

__all__ = ['FederatedMedicalClient']

# server/__init__.py
"""
Federated learning server package
"""

from .federated_server import FederatedServer

__all__ = ['FederatedServer']
