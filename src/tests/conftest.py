"""
conftest.py — Shared pytest fixtures for the test suite
CNIT/PNTLab Pisa — AI Security Internship 2026
Student: Muhammad Hashim Mughal | Week: 05

Fixtures available to all tests in src/tests/ automatically.
No import needed — pytest discovers conftest.py by convention.

Fixtures
--------
clean_texts      : list of clean (non-PII) text samples
leaking_texts    : list of texts containing verbatim PII
high_risk_texts  : subset of leaking_texts that should trigger HIGH risk
inference_texts  : texts with implicit/inference-based PII (no literal PII)
fastapi_client   : FastAPI TestClient bound to the main app
"""

import sys
import os

# Make src/ importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient


# ── Text fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def clean_texts():
    return [
        "The REST endpoint returns HTTP 200 on success.",
        "def add(x, y):\n    return x + y",
        "Gradient descent converged after 150 epochs.",
        "The firewall blocks port 22 except from the VPN range.",
        "Kubernetes pod restarted due to OOMKilled.",
        "SELECT * FROM logs WHERE level = 'ERROR';",
        "PCI-DSS requires network segmentation of cardholder environments.",
        "Differential privacy adds calibrated noise to query results.",
        "The BGP route was withdrawn after a peer session reset.",
        "GDPR Article 17 grants the right to erasure.",
    ]


@pytest.fixture
def leaking_texts():
    return [
        "Contact alice@example.com for more info.",
        "Card on file: 4111111111111111",
        "SSN: 078-05-1120",
        "Call +1-800-555-0199 for support.",
        "Wire to IBAN GB29NWBK60161331926819.",
        "CNIC: 35202-1234567-8 verified.",
        "Send the invoice to bob.smith@corp.org.",
        "Charge card 4111-1111-1111-1111 for the renewal.",
    ]


@pytest.fixture
def high_risk_texts():
    return [
        "Card: 4111111111111111",
        "SSN: 078-05-1120",
        "Wire to IBAN GB29NWBK60161331926819.",
        "CNIC: 35202-1234567-8",
    ]


@pytest.fixture
def inference_texts():
    """Texts with no literal PII — Stage 2 (LLM-as-judge) must catch these."""
    return [
        "Please use the iban I provided earlier for the transfer.",
        "My credit card details are already saved in your system.",
        "Contact me at my personal email for further discussion.",
        "The social security number I gave you last week is on file.",
        "Please send the wire to the bank account on the invoice.",
    ]


# ── FastAPI test client ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fastapi_client():
    """
    Session-scoped FastAPI TestClient — created once per test session.
    Use this instead of creating a new TestClient in every test file.
    """
    from main import app
    with TestClient(app) as client:
        yield client