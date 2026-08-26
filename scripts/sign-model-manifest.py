#!/usr/bin/env python3
"""Create or verify an RSA-PKCS1-SHA256-signed model catalogue envelope.

The private key is supplied externally and is never read from the repository.
Requires the ``cryptography`` package in the release/signing environment.
"""

import argparse
import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def load_private(path: Path) -> rsa.RSAPrivateKey:
    raw = path.read_bytes()
    if not raw.startswith(b"-----"):
        raise ValueError("private key must be PEM-encoded RSA")
    key = serialization.load_pem_private_key(raw, password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("private key is not RSA")
    return key


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sign = sub.add_parser("sign")
    sign.add_argument("payload", type=Path)
    sign.add_argument("output", type=Path)
    sign.add_argument("--private-key", type=Path, required=True)
    sign.add_argument("--catalogue-version", required=True)
    sign.add_argument("--key-id", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("envelope", type=Path)
    verify.add_argument("--public-key", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "sign":
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            parser.error("payload must be a flat catalogue object with a models array")
        signing_bytes = canonical(payload)
        signature = load_private(args.private_key).sign(
            signing_bytes, padding.PKCS1v15(), hashes.SHA256()
        )
        envelope = {
            "catalogue_version": args.catalogue_version,
            "key_id": args.key_id,
            "payload": base64.b64encode(signing_bytes).decode("ascii"),
            "schema_version": 2,
            "signature": base64.b64encode(signature).decode("ascii"),
        }
        args.output.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    else:
        envelope = json.loads(args.envelope.read_text(encoding="utf-8"))
        payload_bytes = base64.b64decode(envelope["payload"], validate=True)
        signature = base64.b64decode(envelope["signature"], validate=True)
        public_raw = args.public_key.read_bytes()
        if public_raw.startswith(b"-----"):
            key = serialization.load_pem_public_key(public_raw)
        else:
            raise ValueError("public key must be PEM-encoded RSA")
        if not isinstance(key, rsa.RSAPublicKey):
            raise ValueError("public key is not RSA")
        key.verify(signature, payload_bytes, padding.PKCS1v15(), hashes.SHA256())
        payload = json.loads(payload_bytes)
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            parser.error("signed payload is not a catalogue")
        print(f"OK schema=2 key_id={envelope['key_id']} version={envelope['catalogue_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
