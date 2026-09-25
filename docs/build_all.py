"""Build the documentation of the monorepo into one site.

The landing page (`mkdocs.yml` at the root) is built into the site's root, and the
documentation of each package with a `mkdocs.yml` into a directory of the site named as
the package's directory, e.g. `site/qiu-signals/`. Every build is strict, so broken
references fail it, and afterwards every relative link of the site must point to a page
or file of it, e.g. the links between the packages, which MkDocs cannot check.

    uv run python docs/build_all.py [--site-dir site]
"""

import argparse
import os
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE_GROUPS = ["packages", "packages-dev", "apps"]


def documented_packages() -> list[Path]:
    """Return the directories of the packages with documentation, sorted by name."""
    return sorted(
        (
            config.parent
            for group in PACKAGE_GROUPS
            for config in (REPOSITORY / group).glob("*/mkdocs.yml")
        ),
        key=lambda package: package.name,
    )


def build(config: Path, site_dir: Path) -> None:
    """Build the documentation of a MkDocs configuration into a directory."""
    print(f"Building {config.relative_to(REPOSITORY)} into {site_dir}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--quiet",
            "-f",
            str(config),
            "-d",
            str(site_dir),
        ],
        check=True,
        cwd=REPOSITORY,
        # MkDocs and Material for MkDocs warn about MkDocs 2 on every build otherwise
        env={
            **os.environ,
            "DISABLE_MKDOCS_2_WARNING": "true",
            "NO_MKDOCS_2_WARNING": "true",
        },
    )


class _LinkParser(HTMLParser):
    """Collects the targets of the `href` and `src` attributes of an HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.targets += [
            value for name, value in attrs if name in ("href", "src") and value
        ]


def site_base_path() -> str:
    """Return the path of the site's deployment, e.g. `/qiu/`, from the landing page."""
    for line in (REPOSITORY / "mkdocs.yml").read_text().splitlines():
        if line.startswith("site_url:"):
            return urlsplit(line.split(":", 1)[1].strip()).path
    return "/"


def broken_links(site_dir: Path, base_path: str = "/") -> list[str]:
    """Return the links of the site's pages to files that do not exist.

    Links relative to a page, and absolute ones under the deployment's `base_path`, e.g.
    those of the 404 pages, are checked; links to other sites are not.
    """
    broken = []
    for page in sorted(site_dir.rglob("*.html")):
        parser = _LinkParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for target in parser.targets:
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:  # external, or within the page
                continue
            path = unquote(url.path)
            if path.startswith("/"):
                if not path.startswith(base_path):
                    broken.append(f"{page.relative_to(site_dir)}: {target}")
                    continue
                resolved = (site_dir / path.removeprefix(base_path)).resolve()
            else:
                resolved = (page.parent / path).resolve()
            if resolved.is_dir():
                resolved = resolved / "index.html"
            if not resolved.is_relative_to(site_dir) or not resolved.exists():
                broken.append(f"{page.relative_to(site_dir)}: {target}")
    return broken


def main() -> None:
    """Build the landing page and the documentation of every package."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    parser.add_argument("--site-dir", type=Path, default=REPOSITORY / "site")
    site_dir = parser.parse_args().site_dir.resolve()

    build(REPOSITORY / "mkdocs.yml", site_dir)
    for package in documented_packages():
        build(package / "mkdocs.yml", site_dir / package.name)

    broken = broken_links(site_dir, site_base_path())
    if broken:
        sys.exit("Broken links:\n" + "\n".join(broken))
    print("All links of the site resolve.")


if __name__ == "__main__":
    main()
