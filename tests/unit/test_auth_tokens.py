"""Spec P8-01 AC1: the JWT access-token verifier of the resource server (design s16-4).

Tokens come from `FakeIssuer`, an in-memory key pair; no identity provider is called.
"""

from __future__ import annotations

import asyncio
import base64
import http.server
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import jwt
import pytest
from pydantic import ValidationError

from rag.config import AuthCallerConfig, ServiceSettings
from rag.security.tokens import JwtTokenVerifier, make_token_verifier
from rag.testing.auth import AUDIENCE, ISSUER, FakeIssuer, new_rsa_key


@pytest.fixture(scope="module")
def issuer() -> FakeIssuer:
    return FakeIssuer()


@pytest.fixture(scope="module")
def verifier(issuer: FakeIssuer) -> JwtTokenVerifier:
    return JwtTokenVerifier.from_jwks(
        issuer.jwks(), issuer=ISSUER, audience=AUDIENCE, algorithms=["RS256"], leeway_s=5
    )


def verify(verifier: JwtTokenVerifier, token: str):  # type: ignore[no-untyped-def]
    return asyncio.run(verifier.verify_token(token))


def test_a_valid_token_gives_its_client_subject_scopes_and_expiry(
    issuer: FakeIssuer, verifier: JwtTokenVerifier
) -> None:
    token = issuer.token(scopes=("rag.local", "openid"), subject="alice", client_id="cli")
    access = verify(verifier, token)
    assert access is not None
    assert access.client_id == "cli" and access.subject == "alice"
    assert access.scopes == ["rag.local", "openid"]
    assert access.expires_at is not None and access.expires_at > time.time()
    assert access.claims is not None and access.claims["iss"] == ISSUER


def test_scp_list_and_azp_are_read_when_scope_and_client_id_are_missing(
    issuer: FakeIssuer, verifier: JwtTokenVerifier
) -> None:
    token = issuer.token(drop=("scope", "client_id"), claims={"scp": ["a", "b"], "azp": "app"})
    access = verify(verifier, token)
    assert access is not None and access.scopes == ["a", "b"] and access.client_id == "app"


def test_an_audience_list_containing_ours_is_accepted(
    issuer: FakeIssuer, verifier: JwtTokenVerifier
) -> None:
    assert verify(verifier, issuer.token(audience=["other", AUDIENCE])) is not None


@pytest.mark.parametrize(
    "case",
    [
        "expired",
        "wrong_audience",
        "wrong_issuer",
        "bad_signature",
        "unknown_kid",
        "no_exp",
        "no_audience",
        "not_a_jwt",
        "alg_none",
        "hs256_with_public_key",
        "not_yet_valid",
    ],
)
def test_bad_tokens_are_refused(issuer: FakeIssuer, verifier: JwtTokenVerifier, case: str) -> None:
    if case == "expired":
        token = issuer.token(expires_in_s=-60)  # beyond the 5 s leeway
    elif case == "wrong_audience":
        token = issuer.token(audience="http://other.example/mcp")
    elif case == "wrong_issuer":
        token = issuer.token(issuer="https://evil.example/")
    elif case == "bad_signature":
        token = issuer.token(key=new_rsa_key())  # same kid, other key
    elif case == "unknown_kid":
        token = issuer.token(kid="nobody")
    elif case == "no_exp":
        token = issuer.token(drop=("exp",))
    elif case == "no_audience":
        token = issuer.token(drop=("aud",))
    elif case == "not_a_jwt":
        token = "not-a-token"
    elif case == "alg_none":
        token = jwt.encode(
            {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300, "scope": "rag.local"},
            key=None,
            algorithm="none",
            headers={"kid": issuer.kid},
        )
    elif case == "hs256_with_public_key":
        # The classic key-confusion attack: an HMAC token keyed with the public key's bytes.
        n = issuer.jwks()["keys"][0]["n"]
        header = {"alg": "HS256", "kid": issuer.kid, "typ": "JWT"}
        body = {"iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 300}

        def part(obj: object) -> str:
            raw = json.dumps(obj).encode()
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

        import hashlib
        import hmac

        signing_input = f"{part(header)}.{part(body)}".encode()
        signature = hmac.new(n.encode(), signing_input, hashlib.sha256).digest()
        token = (
            signing_input.decode() + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        )
    elif case == "not_yet_valid":
        token = issuer.token(claims={"nbf": int(time.time()) + 600})
    else:  # pragma: no cover
        raise AssertionError(case)
    assert verify(verifier, token) is None


def test_the_verifier_never_logs_the_token(
    issuer: FakeIssuer, verifier: JwtTokenVerifier, caplog: pytest.LogCaptureFixture
) -> None:
    token = issuer.token(expires_in_s=-60)
    with caplog.at_level("DEBUG"):
        assert verify(verifier, token) is None
    assert token not in caplog.text and token.split(".")[2] not in caplog.text
    assert "expired" in caplog.text.lower()


@pytest.fixture
def jwks_server(issuer: FakeIssuer) -> Iterator[str]:
    body = json.dumps(issuer.jwks()).encode()
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/jwks.json"
    finally:
        server.shutdown()


def test_keys_from_a_jwks_url(issuer: FakeIssuer, jwks_server: str) -> None:
    settings = _oauth_settings(auth_jwks_url=jwks_server)
    verifier = make_token_verifier(settings)
    assert verify(verifier, issuer.token()) is not None
    assert verify(verifier, issuer.token(key=new_rsa_key())) is None


def test_keys_from_a_jwks_file(issuer: FakeIssuer, tmp_path: Path) -> None:
    path = issuer.write_jwks(tmp_path / "jwks.json")
    verifier = make_token_verifier(_oauth_settings(auth_jwks_path=str(path)))
    assert verify(verifier, issuer.token()) is not None


def _oauth_settings(**overrides: object) -> ServiceSettings:
    values: dict[str, object] = {
        "auth_mode": "oauth",
        "auth_issuer": ISSUER,
        "auth_audience": AUDIENCE,
        "auth_callers": [
            AuthCallerConfig(scope="rag.local", caller="local", acl_groups=["public"])
        ],
        **overrides,
    }
    if "auth_jwks_path" not in values and "auth_jwks_url" not in values:
        values["auth_jwks_path"] = "jwks.json"
    return ServiceSettings(_env_file=None, **values)  # type: ignore[call-arg, arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"auth_issuer": ""},
        {"auth_audience": ""},
        {"auth_callers": []},
        {"auth_algorithms": ["HS256"]},
        {"auth_algorithms": ["none"]},
        {"auth_jwks_path": "a.json", "auth_jwks_url": "http://x/jwks"},
    ],
)
def test_oauth_settings_are_checked(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _oauth_settings(**overrides)


def test_oauth_needs_a_key_source() -> None:
    with pytest.raises(ValidationError):
        ServiceSettings(
            _env_file=None,  # type: ignore[call-arg]
            auth_mode="oauth",
            auth_issuer=ISSUER,
            auth_audience=AUDIENCE,
            auth_callers=[AuthCallerConfig(scope="s", caller="c", acl_groups=["g"])],
        )


def test_local_mode_needs_no_auth_settings() -> None:
    settings = ServiceSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.auth_mode == "local"
