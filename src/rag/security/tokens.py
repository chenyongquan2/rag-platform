"""Bearer-token check of the resource server (spec P8-01, design s16-4).

The service is an OAuth 2.1 resource server: it never issues tokens, it only checks the JWT
access tokens (RFC 9068 shape) that the company's authorization server signed. The MCP SDK
gives the hook (`mcp.server.auth.provider.TokenVerifier`, used by `MCPServer(auth=...,
token_verifier=...)` for `/mcp`); the HTTP API uses the same verifier (`rag.serve.http_app`).

A token is accepted only when all of these hold (PyJWT does the checks):

- the header's `alg` is one of `auth_algorithms` (asymmetric only; `none` and HS* never are),
- the signature verifies with the issuer's key named by `kid` (from a JWKS file or URL),
- `iss` equals `auth_issuer`, `aud` is or contains `auth_audience`,
- `exp` is present and not past, `nbf` (if present) not in the future, within `auth_leeway_s`.

Scopes come from `scope` (space separated, RFC 9068) or `scp` (a list, as some providers
write it); the client from `client_id`, else `azp`, else `sub`. The header's `typ` is not
checked: RFC 9068 asks for `at+jwt`, but several providers send `JWT`; audience and issuer
already tie the token to this service. A refused token gives `None`
(the SDK and the HTTP layer answer 401) and one log line with the reason, never the token.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import anyio
import jwt
from jwt import PyJWK, PyJWKClient, PyJWKSet
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.routes import build_resource_metadata_url
from pydantic import AnyHttpUrl

from rag.config import ServiceSettings

logger = logging.getLogger(__name__)

_REQUIRED_CLAIMS = ("exp", "iss", "aud")


class KeySource(Protocol):
    def key_for(self, kid: str | None) -> PyJWK | None: ...


class StaticKeys:
    """Keys of a JWKS document (a file read at start). Without `kid` only a set of one key
    can be used."""

    def __init__(self, jwks: Mapping[str, Any]) -> None:
        self._set = PyJWKSet.from_dict(dict(jwks))

    @classmethod
    def from_file(cls, path: str | Path) -> StaticKeys:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def key_for(self, kid: str | None) -> PyJWK | None:
        keys = self._set.keys
        if kid is None:
            return keys[0] if len(keys) == 1 else None
        return next((k for k in keys if k.key_id == kid), None)


class RemoteKeys:
    """Keys fetched from the issuer's JWKS URL with `PyJWKClient`. Only the whole key set is
    cached, for `cache_s` seconds: a key the issuer removed is refused at the latest `cache_s`
    after its removal. PyJWT's per-kid key cache (`cache_keys`) stays off, because it keeps a
    key without any time limit (review P8-01 round 1). An unknown `kid` fetches the set again,
    at most once per `cooldown_s` seconds (PyJWT's `cooldown_duration`), so tokens with made-up
    kids cannot make the service hammer the issuer; a newly rotated key can wait that long."""

    def __init__(
        self, url: str, *, cache_s: int = 300, cooldown_s: float = 30.0, timeout_s: float = 5.0
    ) -> None:
        self._client = PyJWKClient(
            url,
            cache_keys=False,
            cache_jwk_set=True,
            lifespan=cache_s,
            cooldown_duration=cooldown_s,
            timeout=timeout_s,
        )

    def key_for(self, kid: str | None) -> PyJWK | None:
        if kid is None:
            keys = self._client.get_signing_keys()
            return keys[0] if len(keys) == 1 else None
        return self._client.get_signing_key(kid)


def _scopes(claims: Mapping[str, Any]) -> list[str]:
    scope = claims.get("scope")
    if isinstance(scope, str):
        return [s for s in scope.split(" ") if s]
    scp = claims.get("scp")
    if isinstance(scp, list):
        return [s for s in scp if isinstance(s, str) and s]
    if isinstance(scp, str):
        return [s for s in scp.split(" ") if s]
    return []


class JwtTokenVerifier:
    """`TokenVerifier` for JWT access tokens (module doc)."""

    def __init__(
        self,
        keys: KeySource,
        *,
        issuer: str,
        audience: str,
        algorithms: Sequence[str],
        leeway_s: int = 30,
    ) -> None:
        if not issuer or not audience:
            raise ValueError("issuer and audience are required")
        self._keys = keys
        self.issuer = issuer
        self.audience = audience
        self.algorithms = list(algorithms)
        self.leeway_s = leeway_s

    @classmethod
    def from_jwks(cls, jwks: Mapping[str, Any], **kwargs: Any) -> JwtTokenVerifier:
        return cls(StaticKeys(jwks), **kwargs)

    async def verify_token(self, token: str) -> AccessToken | None:
        # The key lookup may fetch the JWKS over the network: not on the event loop.
        return await anyio.to_thread.run_sync(self.verify_sync, token)

    def verify_sync(self, token: str) -> AccessToken | None:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            logger.info("bearer token refused: unreadable header (%s)", type(exc).__name__)
            return None
        alg = header.get("alg")
        if alg not in self.algorithms:
            logger.info("bearer token refused: algorithm %r not allowed", alg)
            return None
        kid = header.get("kid")
        try:
            key = self._keys.key_for(kid if isinstance(kid, str) else None)
        except Exception as exc:
            # The JWKS could not be fetched or read (network, not JSON, no usable key): the
            # token cannot be checked, so it is refused (401), never a 500.
            logger.warning("bearer token refused: key lookup failed (%s)", type(exc).__name__)
            return None
        if key is None:
            logger.info("bearer token refused: no key for kid %r", kid)
            return None
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[alg],
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_s,
                options={"require": list(_REQUIRED_CLAIMS)},
            )
        except jwt.ExpiredSignatureError:
            logger.info("bearer token refused: expired")
            return None
        except jwt.PyJWTError as exc:
            # The message names the failed check (audience, issuer, signature), not the token.
            logger.info("bearer token refused: %s (%s)", type(exc).__name__, exc)
            return None
        subject = claims.get("sub") if isinstance(claims.get("sub"), str) else None
        client_id = next(
            (
                claims[name]
                for name in ("client_id", "azp")
                if isinstance(claims.get(name), str) and claims[name]
            ),
            subject or "",
        )
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=_scopes(claims),
            expires_at=int(claims["exp"]),
            resource=self.audience,
            subject=subject,
            claims={"iss": claims["iss"]},
        )


def resource_url(settings: ServiceSettings) -> str:
    """The protected resource's public URL (the MCP endpoint): `auth_resource_url`, else the
    token audience."""
    return str(settings.auth_resource_url or settings.auth_audience)


def protected_resource_metadata_url(settings: ServiceSettings) -> str | None:
    """RFC 9728 metadata URL of the resource, which the 401 challenge points to."""
    url = resource_url(settings)
    if not url:
        return None
    return str(build_resource_metadata_url(AnyHttpUrl(url)))


def make_token_verifier(settings: ServiceSettings) -> JwtTokenVerifier:
    """The verifier the settings describe (`auth_mode=oauth`)."""
    if settings.auth_jwks_url:
        keys: KeySource = RemoteKeys(
            settings.auth_jwks_url,
            cache_s=settings.auth_jwks_cache_s,
            cooldown_s=settings.auth_jwks_refetch_cooldown_s,
        )
    else:
        keys = StaticKeys.from_file(settings.auth_jwks_path)
    return JwtTokenVerifier(
        keys,
        issuer=settings.auth_issuer,
        audience=settings.auth_audience,
        algorithms=settings.auth_algorithms,
        leeway_s=settings.auth_leeway_s,
    )
