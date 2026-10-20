"""CHM folder source, unpack step and the `chm_dir` connector (spec P4-01 AC4, AC5; s5-1, s5-2)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rag.config import ServiceSettings
from rag.data import SourceDocument
from rag.data.chm import (
    ChmError,
    DirectoryChmSource,
    SevenZipUnpacker,
    UnpackError,
    charset_hint_for_lcid,
    find_unpacker,
    unpack_chm,
)
from rag.data.chm.__main__ import main as chm_main
from rag.registry import connectors
from rag.testing.chm import h1, hhc, hhk, page_html, text, write_volume

EN_TREE = [
    ("Frobnitz SDK", "index.htm", []),
    ("Queue", "queue.htm", [("Add", "queue_add.htm", []), ("Clear", "queue_clear.htm", [])]),
    ("English Only", "only_en.htm", []),
]
ZH_TREE = [
    ("Frobnitz SDK", "index.htm", []),
    ("队列", "queue.htm", [("添加", "queue_add.htm", []), ("清空", "queue_clear.htm", [])]),
]


def en_pages() -> dict[str, bytes]:
    def p(title: str, body: str) -> bytes:
        return page_html(h1(title), text(body), title=title).encode("cp1252")

    return {
        "index.htm": p("Frobnitz SDK", "The Frobnitz SDK moves widgets."),
        "queue.htm": p("Queue", "A queue holds widgets."),
        "queue_add.htm": p("Queue::Add", "Add a widget to the queue."),
        "queue_clear.htm": p("Queue::Clear", "Remove every widget."),
        "only_en.htm": p("Only English", "This page has no translation."),
    }


def zh_pages() -> dict[str, bytes]:
    def p(title: str, body: str) -> bytes:
        return page_html(h1(title), text(body), title=title, charset="gb2312").encode("gb18030")

    return {
        "index.htm": p("Frobnitz SDK", "Frobnitz SDK 用来移动部件。"),
        "queue.htm": p("队列", "队列保存部件。"),
        "queue_add.htm": p("Queue::Add", "把部件添加到队列。"),
        "queue_clear.htm": p("Queue::Clear", "移除所有部件。"),
    }


@pytest.fixture
def manual(tmp_path: Path) -> dict[str, Path]:
    en = write_volume(
        tmp_path / "en",
        en_pages(),
        toc=("api.hhc", hhc(EN_TREE)),
        index=("api.hhk", hhk([("Add", ["queue_add.htm"])])),
        lcid=0x409,
    )
    zh = write_volume(
        tmp_path / "zh",
        zh_pages(),
        toc=("api_chinese.hhc", hhc(ZH_TREE)),
        index=("api_chinese.hhk", hhk([("添加", ["queue_add.htm"])])),
        toc_encoding="gb18030",
        lcid=0x804,
    )
    return {"en": en, "zh": zh}


def make_connector(manual: dict[str, Path], **extra: Any) -> Any:
    volumes = [{"language": lang, "path": str(path)} for lang, path in manual.items()]
    return connectors.create(
        "chm_dir", kb="mt5", settings=ServiceSettings(), volumes=volumes, **extra
    )


# --- the folder source ------------------------------------------------------------------------


def test_directory_source_lists_pages_and_skips_chm_internals(manual: dict[str, Path]) -> None:
    source = DirectoryChmSource(manual["en"])
    assert source.names() == [
        "index.htm",
        "only_en.htm",
        "queue.htm",
        "queue_add.htm",
        "queue_clear.htm",
    ]


def test_directory_source_finds_subfolders_and_chinese_names(tmp_path: Path) -> None:
    raw = page_html(h1("x")).encode("cp1252")
    root = write_volume(tmp_path / "v", {"sub/deep.htm": raw, "中文页.htm": raw, "UP.HTML": raw})
    assert DirectoryChmSource(root).names() == ["UP.HTML", "sub/deep.htm", "中文页.htm"]


def test_directory_source_reads_bytes_and_refuses_paths_outside_the_folder(
    manual: dict[str, Path],
) -> None:
    source = DirectoryChmSource(manual["zh"])
    assert source.read("queue.htm") == (manual["zh"] / "queue.htm").read_bytes()
    for bad in ("../en/queue.htm", "/etc/passwd", "C:/x.htm"):
        with pytest.raises(ChmError):
            source.read(bad)
    with pytest.raises(ChmError, match=r"missing\.htm"):
        source.read("missing.htm")


def test_directory_source_finds_toc_and_index_by_extension(manual: dict[str, Path]) -> None:
    en = DirectoryChmSource(manual["en"])
    assert b"Frobnitz SDK" in (en.toc() or b"")
    assert b"Add" in (en.index() or b"")
    assert DirectoryChmSource(manual["zh"]).toc_name == "api_chinese.hhc"


def test_directory_source_without_toc_or_with_two_tocs(tmp_path: Path) -> None:
    raw = page_html(h1("x")).encode("cp1252")
    none = write_volume(tmp_path / "a", {"a.htm": raw})
    assert DirectoryChmSource(none).toc() is None
    two = write_volume(tmp_path / "b", {"a.htm": raw}, toc=("one.hhc", hhc([])))
    (two / "two.hhc").write_text(hhc([]), encoding="ascii")
    with pytest.raises(ChmError, match=r"two\.hhc"):
        DirectoryChmSource(two).toc()
    assert DirectoryChmSource(two, toc="one.hhc").toc() is not None


def test_directory_source_reads_locale_and_title_from_system_file(
    manual: dict[str, Path], tmp_path: Path
) -> None:
    zh = DirectoryChmSource(manual["zh"]).info()
    en = DirectoryChmSource(manual["en"]).info()
    assert (zh.lcid, en.lcid) == (0x804, 0x409)
    assert zh.title == "Frobnitz SDK - User Guide"
    assert zh.charset_hint == "gb18030"
    assert en.charset_hint == "cp1252"
    bare = write_volume(tmp_path / "bare", {"a.htm": b"<html></html>"}, internals=False)
    assert DirectoryChmSource(bare).info().lcid is None
    assert DirectoryChmSource(bare).info().charset_hint is None


def test_charset_hint_for_lcid() -> None:
    assert charset_hint_for_lcid(0x804) == "gb18030"  # Chinese (PRC)
    assert charset_hint_for_lcid(0x409) == "cp1252"  # English (US)
    assert charset_hint_for_lcid(0x40C) == "cp1252"  # French
    assert charset_hint_for_lcid(0x419) is None  # Russian: another family, no guess
    assert charset_hint_for_lcid(None) is None


# --- the unpack step --------------------------------------------------------------------------


class FakeUnpacker:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def unpack(self, chm: Path, out_dir: Path) -> None:
        self.calls.append((chm, out_dir))
        write_volume(out_dir, en_pages(), toc=("api.hhc", hhc(EN_TREE)), lcid=0x409)


def make_chm(tmp_path: Path) -> Path:
    chm = tmp_path / "Manual.chm"
    chm.write_bytes(b"ITSF" + b"\x00" * 64)
    return chm


def test_unpack_runs_the_tool_and_writes_a_manifest(tmp_path: Path) -> None:
    chm, fake = make_chm(tmp_path), FakeUnpacker()
    out = tmp_path / "out"
    result = unpack_chm(chm, out, unpacker=fake)
    assert fake.calls == [(chm, out)]
    assert result.tool == "fake"
    assert result.pages == 5
    assert result.chm_sha256 == hashlib.sha256(chm.read_bytes()).hexdigest()
    manifest = json.loads((out / "_unpack.json").read_text("utf-8"))
    assert manifest["chm_name"] == "Manual.chm"
    assert manifest["chm_bytes"] == chm.stat().st_size
    assert manifest["pages"] == 5
    assert manifest["tool"] == "fake"
    # no folder names of this machine in it (the temp folder's own name is distinctive)
    assert "/" not in manifest["chm_name"] and tmp_path.name not in json.dumps(manifest)


def test_unpack_refuses_a_folder_that_already_has_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(UnpackError, match="not empty"):
        unpack_chm(make_chm(tmp_path), out, unpacker=FakeUnpacker())
    assert (out / "keep.txt").exists()
    unpack_chm(make_chm(tmp_path), out, unpacker=FakeUnpacker(), overwrite=True)
    assert (out / "queue.htm").exists()


def test_unpack_missing_chm_and_empty_result(tmp_path: Path) -> None:
    with pytest.raises(UnpackError, match="no such file"):
        unpack_chm(tmp_path / "none.chm", tmp_path / "o", unpacker=FakeUnpacker())

    class Silent:
        name = "silent"

        def unpack(self, chm: Path, out_dir: Path) -> None:
            out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(UnpackError, match="no pages"):
        unpack_chm(make_chm(tmp_path), tmp_path / "o2", unpacker=Silent())


def test_seven_zip_unpacker_builds_the_command_and_reports_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Everything is Ok", stderr="")

    monkeypatch.setattr("rag.data.chm.unpack.subprocess.run", fake_run)
    chm, out = tmp_path / "a.chm", tmp_path / "o"
    SevenZipUnpacker("7z").unpack(chm, out)
    assert seen == [["7z", "x", "-y", f"-o{out}", str(chm)]]

    def failing(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="Can not open the file")

    monkeypatch.setattr("rag.data.chm.unpack.subprocess.run", failing)
    with pytest.raises(UnpackError, match=r"exit code 2.*Can not open"):
        SevenZipUnpacker("7z").unpack(chm, out)


def test_find_unpacker_prefers_seven_zip_then_chmlib_then_names_what_it_tried() -> None:
    found = {"7z": "/usr/bin/7z", "extract_chmLib": "/usr/bin/extract_chmLib"}
    assert find_unpacker(which=lambda n: found.get(n), exists=lambda p: False).name == "7z"
    found.pop("7z")
    assert (
        find_unpacker(which=lambda n: found.get(n), exists=lambda p: False).name == "extract_chmLib"
    )
    with pytest.raises(UnpackError) as err:
        find_unpacker(which=lambda n: None, exists=lambda p: False)
    assert "7z" in str(err.value) and "extract_chmLib" in str(err.value)


def test_find_unpacker_uses_the_windows_default_install_path() -> None:
    wanted = "C:/Program Files/7-Zip/7z.exe"
    unpacker = find_unpacker(which=lambda n: None, exists=lambda p: p.replace("\\", "/") == wanted)
    assert unpacker.name == "7z"


def test_chm_cli_unpack_prints_a_summary_without_page_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    chm = make_chm(tmp_path)
    code = chm_main(
        ["unpack", "--chm", str(chm), "--out", str(tmp_path / "o")], unpacker=FakeUnpacker()
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "pages: 5" in out and "fake" in out
    assert (
        chm_main(
            ["unpack", "--chm", str(tmp_path / "x.chm"), "--out", str(tmp_path / "p")],
            unpacker=FakeUnpacker(),
        )
        == 2
    )


# --- the connector ----------------------------------------------------------------------------


def docs_of(manual: dict[str, Path], **extra: Any) -> dict[str, SourceDocument]:
    return {d.doc_id: d for d in make_connector(manual, **extra).documents()}


def test_connector_gives_ids_language_page_key_and_title_path(manual: dict[str, Path]) -> None:
    docs = docs_of(manual)
    assert len(docs) == 9  # 5 English + 4 Chinese pages
    en = docs["mt5:en/queue_add.htm"]
    zh = docs["mt5:zh/queue_add.htm"]
    assert (en.extra["language"], zh.extra["language"]) == ("en", "zh")
    assert en.extra["page_key"] == zh.extra["page_key"] == "queue_add"
    assert en.extra["toc_path"] == ["Queue", "Add"]
    assert zh.extra["toc_path"] == ["队列", "添加"]
    assert en.source_rel == "en/queue_add.htm"
    assert en.ext == ".htm" and en.kb == "mt5" and en.source_url is None
    assert zh.extra["charset_hint"] == "gb18030"
    assert zh.content == (manual["zh"] / "queue_add.htm").read_bytes()  # raw bytes, undecoded


def test_connector_attaches_index_keywords(manual: dict[str, Path]) -> None:
    docs = docs_of(manual)
    assert docs["mt5:en/queue_add.htm"].extra["keywords"] == ["Add"]
    assert docs["mt5:zh/queue_add.htm"].extra["keywords"] == ["添加"]
    assert docs["mt5:en/queue.htm"].extra["keywords"] == []


def test_connector_languages_have_one_page_key_each_and_the_english_only_page_is_unpaired(
    manual: dict[str, Path],
) -> None:
    docs = docs_of(manual)
    keys = {
        lang: {d.extra["page_key"] for d in docs.values() if d.extra["language"] == lang}
        for lang in ("en", "zh")
    }
    assert keys["zh"] < keys["en"]
    assert keys["en"] - keys["zh"] == {"only_en"}


def test_connector_source_version_is_the_chm_hash_or_the_page_hash(
    manual: dict[str, Path], tmp_path: Path
) -> None:
    docs = docs_of(manual)
    page = (manual["en"] / "queue.htm").read_bytes()
    assert (
        docs["mt5:en/queue.htm"].source_version == f"sha256:{hashlib.sha256(page).hexdigest()[:16]}"
    )
    (manual["en"] / "_unpack.json").write_text(
        json.dumps({"chm_sha256": "ab" * 32, "chm_name": "x.chm"}), encoding="utf-8"
    )
    docs = docs_of(manual)
    assert docs["mt5:en/queue.htm"].source_version == "chm:" + "ab" * 8
    assert docs["mt5:zh/queue.htm"].source_version.startswith("sha256:")


def test_connector_limit_is_per_volume_and_keeps_pairs(manual: dict[str, Path]) -> None:
    docs = docs_of(manual, limit=2)
    assert sorted(docs) == [
        "mt5:en/index.htm",
        "mt5:en/only_en.htm",
        "mt5:zh/index.htm",
        "mt5:zh/queue.htm",
    ]
    with pytest.raises(ValueError, match="limit"):
        make_connector(manual, limit=0)


def test_connector_sections_select_the_pages_under_top_level_contents_entries(
    manual: dict[str, Path],
) -> None:
    volumes = [
        {"language": "en", "path": str(manual["en"]), "sections": ["Queue"]},
        {"language": "zh", "path": str(manual["zh"]), "sections": ["队列", "Frobnitz SDK"]},
    ]
    connector = connectors.create("chm_dir", kb="mt5", settings=ServiceSettings(), volumes=volumes)
    docs = sorted(d.doc_id for d in connector.documents())
    assert docs == [
        "mt5:en/queue.htm",
        "mt5:en/queue_add.htm",
        "mt5:en/queue_clear.htm",
        "mt5:zh/index.htm",
        "mt5:zh/queue.htm",
        "mt5:zh/queue_add.htm",
        "mt5:zh/queue_clear.htm",
    ]
    assert connector.volume_stats["en"].pages == 5  # the volume's own count does not change
    assert connector.volume_stats["en"].selected == 3


def test_connector_sections_leave_out_pages_that_are_not_in_the_contents(
    manual: dict[str, Path],
) -> None:
    write_volume(
        manual["en"], {"orphan.htm": page_html(h1("Orphan")).encode("cp1252")}, internals=False
    )
    volumes = [{"language": "en", "path": str(manual["en"]), "sections": ["Queue"]}]
    connector = connectors.create("chm_dir", kb="mt5", settings=ServiceSettings(), volumes=volumes)
    assert "mt5:en/orphan.htm" not in {d.doc_id for d in connector.documents()}


def test_connector_sections_with_a_wrong_name_name_the_known_ones(manual: dict[str, Path]) -> None:
    volumes = [{"language": "en", "path": str(manual["en"]), "sections": ["Queu"]}]
    connector = connectors.create("chm_dir", kb="mt5", settings=ServiceSettings(), volumes=volumes)
    with pytest.raises(ChmError, match=r"Queu.*known.*Queue"):
        list(connector.documents())
    for bad in ("Queue", [], [1]):
        with pytest.raises(ValueError, match="sections"):
            connectors.create(
                "chm_dir",
                kb="mt5",
                settings=ServiceSettings(),
                volumes=[{"language": "en", "path": str(manual["en"]), "sections": bad}],
            )


def test_connector_reports_toc_and_page_mismatches(manual: dict[str, Path]) -> None:
    (manual["zh"] / "queue_clear.htm").unlink()  # in the toc, no file
    write_volume(
        manual["zh"],
        {"orphan.htm": page_html(h1("Orphan")).encode("cp1252")},
        internals=False,
    )  # a file, not in the toc
    connector = make_connector(manual)
    list(connector.documents())
    zh = connector.volume_stats["zh"]
    assert zh.pages == 4 and zh.in_toc == 3 and zh.not_in_toc == ["orphan"]
    assert zh.toc_without_page == ["queue_clear"]
    orphan = next(d for d in connector.documents() if d.doc_id == "mt5:zh/orphan.htm")
    assert orphan.extra["toc_path"] is None


def test_connector_chinese_file_names_get_stable_ids_and_keys(tmp_path: Path) -> None:
    raw = page_html(h1("x"), charset="gb2312").encode("gb18030")
    root = write_volume(
        tmp_path / "z",
        {"中文页.htm": raw},
        toc=("a.hhc", hhc([("中文", "%E4%B8%AD%E6%96%87%E9%A1%B5.htm", [])])),
        toc_encoding="gb18030",
    )
    connector = connectors.create(
        "chm_dir",
        kb="mt5",
        settings=ServiceSettings(),
        volumes=[{"language": "zh", "path": str(root)}],
    )
    (doc,) = list(connector.documents())
    assert doc.doc_id == "mt5:zh/中文页.htm"
    assert doc.extra["page_key"] == "中文页"
    assert doc.extra["toc_path"] == ["中文"]


def test_connector_checks_its_configuration(manual: dict[str, Path], tmp_path: Path) -> None:
    base = {"kb": "mt5", "settings": ServiceSettings()}
    with pytest.raises(ValueError, match="volumes"):
        connectors.create("chm_dir", **base, volumes=[])
    with pytest.raises(ValueError, match="twice"):
        connectors.create(
            "chm_dir",
            **base,
            volumes=[
                {"language": "en", "path": str(manual["en"])},
                {"language": "en", "path": str(manual["zh"])},
            ],
        )
    with pytest.raises(ChmError, match="does not exist"):
        connectors.create(
            "chm_dir", **base, volumes=[{"language": "en", "path": str(tmp_path / "none")}]
        )
    with pytest.raises(ValueError, match="language"):
        connectors.create("chm_dir", **base, volumes=[{"path": str(manual["en"])}])


def test_source_document_extra_defaults_to_empty_and_other_connectors_are_unaffected() -> None:
    doc = SourceDocument(
        doc_id="k:a.md",
        kb="k",
        source_rel="a.md",
        source_url=None,
        ext=".md",
        content=b"x",
        source_version="v",
    )
    assert doc.extra == {}
