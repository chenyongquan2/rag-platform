"""A stand-in authorization server for tests (spec P8-01): it only signs access tokens.

`FakeIssuer` makes an RSA key pair in memory, publishes the public key as a JWKS (JSON Web Key
Set) and signs RFC 9068 style JWT access tokens. Nothing is written to disk unless a test asks
(`write_jwks`), and the private key never leaves the object. Tests make bad tokens with the
keyword arguments: an expired one, another audience or issuer, a key the JWKS does not know.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

ISSUER = "https://idp.example.test/"
AUDIENCE = "http://127.0.0.1:28080/mcp"


def new_rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeIssuer:
    """Signs tokens with one RSA key (`kid`); `jwks()` is the public half."""

    def __init__(
        self, *, issuer: str = ISSUER, audience: str = AUDIENCE, kid: str = "test-key-1"
    ) -> None:
        self.issuer = issuer
        self.audience = audience
        self.kid = kid
        self._key = new_rsa_key()

    def jwks(self) -> dict[str, Any]:
        jwk = json.loads(RSAAlgorithm.to_jwk(self._key.public_key()))
        return {"keys": [{**jwk, "kid": self.kid, "use": "sig", "alg": "RS256"}]}

    def write_jwks(self, path: Path) -> Path:
        path.write_text(json.dumps(self.jwks()), encoding="utf-8", newline="")
        return path

    def token(
        self,
        *,
        scopes: Sequence[str] = ("rag.local",),
        subject: str = "user-1",
        client_id: str = "client-1",
        expires_in_s: int = 300,
        issuer: str | None = None,
        audience: str | Sequence[str] | None = None,
        key: rsa.RSAPrivateKey | None = None,
        kid: str | None = None,
        claims: Mapping[str, Any] | None = None,
        drop: Sequence[str] = (),
    ) -> str:
        """A signed access token. `key` signs with another key (a bad signature under this
        issuer's `kid`); `drop` removes claims; `claims` adds or replaces some."""
        now = int(time.time())
        body: dict[str, Any] = {
            "iss": self.issuer if issuer is None else issuer,
            "aud": self.audience if audience is None else audience,
            "sub": subject,
            "client_id": client_id,
            "scope": " ".join(scopes),
            "iat": now,
            "exp": now + expires_in_s,
            "jti": uuid.uuid4().hex,
        }
        body.update(claims or {})
        for name in drop:
            body.pop(name, None)
        return jwt.encode(
            body,
            key or self._key,
            algorithm="RS256",
            headers={"kid": self.kid if kid is None else kid, "typ": "at+jwt"},
        )
