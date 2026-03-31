"""
algos/facmac — FACMAC Algorithm Stub
=====================================

FACMAC: Factored Multi-Agent Centralized Policy Gradients
Paper: arxiv:2201.06233 (NeurIPS 2022)

Exports:
    FACMACPolicy
    FACMACTrainer
"""

from algos.facmac.policy import FACMACPolicy
from algos.facmac.trainer import FACMACTrainer

__all__ = ["FACMACPolicy", "FACMACTrainer"]
