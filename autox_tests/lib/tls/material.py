"""Short-lived, session-specific CA and server certificates for TLS probes."""

from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
import subprocess


def generate_material() -> dict[str, str]:
    """Return PEM material; private keys are only mounted through a test Secret."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    now = datetime.now(timezone.utc)

    def key():
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def name(label):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, label)])

    def certificate(
        subject,
        issuer,
        public_key,
        signer,
        *,
        ca=False,
        expired=False,
        wrong_host=False,
    ):
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=2))
            .not_valid_after(
                now - timedelta(days=1) if expired else now + timedelta(days=2)
            )
            .add_extension(
                x509.BasicConstraints(ca=ca, path_length=None), critical=True
            )
        )
        if ca:
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=False,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            ).add_extension(
                x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False
            )
        else:
            builder = builder.add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(signer.public_key()),
                critical=False,
            )
            san = (
                [x509.DNSName("wrong.invalid")]
                if wrong_host
                else [
                    x509.IPAddress(ip_address("127.0.0.1")),
                    x509.DNSName("localhost"),
                ]
            )
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san), critical=False
            )
        return (
            builder.sign(signer, hashes.SHA256())
            .public_bytes(serialization.Encoding.PEM)
            .decode()
        )

    root_key, other_key, server_key = key(), key(), key()
    root, other = name("autox-session-ca"), name("autox-untrusted-ca")
    result = {
        "ca.pem": certificate(root, root, root_key.public_key(), root_key, ca=True),
        "other.pem": certificate(
            other, other, other_key.public_key(), other_key, ca=True
        ),
        "server.key": server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
    }
    for label, options in (
        ("server", {}),
        ("expired", {"expired": True}),
        ("wrong-host", {"wrong_host": True}),
    ):
        result[f"{label}.pem"] = certificate(
            name("autox-server"), root, server_key.public_key(), root_key, **options
        )
    result["ca.hash"] = subprocess.run(
        ["openssl", "x509", "-subject_hash", "-noout"],
        input=result["ca.pem"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return result
