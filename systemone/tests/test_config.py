from ipaddress import IPv4Address

import pytest
from pydantic import ValidationError

from r9v_systemone.config import Settings


def test_settings_default_to_local_r9v_and_serial_execution(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_R9V_BASE_URL", raising=False)
    monkeypatch.delenv("SYSTEMONE_CONCURRENCY", raising=False)
    monkeypatch.delenv("SYSTEMONE_BIND_HOST", raising=False)

    settings = Settings()

    assert str(settings.r9v_base_url) == "http://127.0.0.1:8000/"
    assert settings.concurrency == 1
    assert settings.bind_host == IPv4Address("127.0.0.1")


@pytest.mark.parametrize("bind_host", ["0.0.0.0", "192.0.2.1", "::"])
def test_settings_reject_non_loopback_bind_addresses(bind_host):
    with pytest.raises(ValidationError):
        Settings(bind_host=bind_host)


def test_settings_accept_loopback_ipv6():
    assert Settings(bind_host="::1").bind_host.is_loopback


def test_settings_are_immutable():
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.concurrency = 2


def test_settings_reject_coerced_constructor_values_and_unknown_fields():
    with pytest.raises(ValidationError):
        Settings(concurrency="2")
    with pytest.raises(ValidationError):
        Settings(unexpected=True)
