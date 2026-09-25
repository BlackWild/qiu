"""Generate the API reference pages of a package from the modules of its `src`.

Run by the `gen-files` plugin of MkDocs for the package whose `mkdocs.yml` is built
(see `docs/mkdocs.base.yml`): one page per module under `reference/`, a package's page
being its `__init__` module, and the navigation of them in `reference/SUMMARY.md`.
Private modules, starting with `_`, are skipped.
"""

from pathlib import Path

import mkdocs_gen_files

package_root = Path(mkdocs_gen_files.config["config_file_path"]).parent
source_root = package_root / "src"

navigation = mkdocs_gen_files.Nav()  # type: ignore[attr-defined]

for path in sorted(source_root.rglob("*.py")):
    module_path = path.relative_to(source_root).with_suffix("")
    doc_path = path.relative_to(source_root).with_suffix(".md")
    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    elif parts[-1].startswith("_"):
        continue

    navigation[parts] = doc_path.as_posix()
    with mkdocs_gen_files.open(Path("reference", doc_path), "w") as page:
        page.write(f"::: {'.'.join(parts)}\n")

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as summary:
    summary.writelines(navigation.build_literate_nav())
