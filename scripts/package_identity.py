#!/usr/bin/env python3
"""Package identity of an .msix, and whether that exact version is installed.

#603: installing the same package version in place, over and over, once left
the console dropping every inbound connection on the LAN API port while the
app logged "listening". A newer revision fixed it. deploy.sh therefore does
not reinstall a version that is already registered, unless asked to.

  package_identity.py identity <package.msix>
      prints "<Name> <Version>" from the AppxManifest.xml inside the package
  package_identity.py installed <package.msix>  < packagemanager/packages JSON
      prints the PackageFullName of the registered package with that Name and
      Version, or nothing; exit 0 either way

Exit 2 when the package or the listing cannot be read. Read from the manifest,
not the file name, so a renamed .msix is still recognised.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

FOUNDATION = "{http://schemas.microsoft.com/appx/manifest/foundation/windows10}"


class IdentityError(Exception):
    pass


def msix_identity(path: Path) -> tuple[str, str]:
    """(Name, Version) of the package's Identity element."""
    try:
        with zipfile.ZipFile(path) as package:
            manifest = package.read("AppxManifest.xml")
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise IdentityError(f"cannot read AppxManifest.xml from {path}: {error}") from error
    try:
        identity = ET.fromstring(manifest).find(f"{FOUNDATION}Identity")
    except ET.ParseError as error:
        raise IdentityError(f"{path}: AppxManifest.xml is not XML: {error}") from error
    if identity is None or not identity.get("Name") or not identity.get("Version"):
        raise IdentityError(f"{path}: AppxManifest.xml has no Identity Name and Version")
    return identity.get("Name"), identity.get("Version")


def installed_full_name(listing: dict, name: str, version: str) -> str | None:
    """The registered PackageFullName with this Name and Version, if any.

    A full name is Name_Version_Architecture_ResourceId_PublisherId, so the
    prefix "Name_Version_" names exactly one package version of one family.
    """
    prefix = f"{name}_{version}_"
    for package in listing.get("InstalledPackages", []):
        full_name = package.get("PackageFullName", "")
        if full_name.startswith(prefix):
            return full_name
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("identity", "installed"):
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        name, version = msix_identity(Path(argv[2]))
        if argv[1] == "identity":
            print(f"{name} {version}")
            return 0
        try:
            listing = json.load(sys.stdin)
        except json.JSONDecodeError as error:
            raise IdentityError(f"the package listing is not JSON: {error}") from error
        full_name = installed_full_name(listing, name, version)
        if full_name:
            print(full_name)
        return 0
    except IdentityError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
