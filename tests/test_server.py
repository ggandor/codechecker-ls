import pathlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_lsp
from codechecker_ls.server import CODECHECKER_DIAGNOSTIC
from lsprotocol import types as lsp_types
from lsprotocol.types import (
    TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS,
    ClientCapabilities,
    Diagnostic,
    DiagnosticSeverity,
    DocumentDiagnosticParams,
    InitializeParams,
    Position,
    Range,
    RelatedFullDocumentDiagnosticReport,
    TextDocumentIdentifier,
    TextDocumentItem,
    WorkDoneProgressBegin,
    WorkDoneProgressEnd,
    WorkDoneProgressReport,
    WorkspaceDiagnosticParams,
    WorkspaceFullDocumentDiagnosticReport,
)
from pytest_lsp import ClientServerConfig, LanguageClient


TEST_ROOT = pathlib.Path(__file__).parent
SOURCES_ROOT = TEST_ROOT / 'sources'
PROJECT_ROOT = TEST_ROOT.parent
CODECHECKER_LS_PATH = PROJECT_ROOT / 'codechecker_ls' / 'server.py'


def have_same_diagnostics(expected, result):
    if len(expected) != len(result):
        return False

    def sort_key(d):
        return d.range.start.line, d.range.start.character, d.message

    # Order should not matter.
    expected.sort(key=sort_key)
    result.sort(key=sort_key)

    for exp, res in zip(expected, result):
        if not (
            exp.message == res.message
            and exp.severity == res.severity
            # XXX: Workaround until the (de)serialization problem with
            # `codechecker/diagnostic` is solved (we get plain Objects
            # back).
            and exp.range.start.line == res.range.start.line
            and exp.range.start.character == res.range.start.character
            and exp.range.end.line == res.range.end.line
            and exp.range.end.character == res.range.end.character
        ):
            return False

    return True


def expected_diagnostics() -> dict[str, list[Diagnostic]]:
    return {
        'src/main.c': [
            Diagnostic(
                range=Range(
                    start=Position(line=22, character=4),
                    end=Position(line=22, character=4),
                ),
                message="Undefined or garbage value returned to caller",
                severity=DiagnosticSeverity.Error,
            ),
            Diagnostic(
                range=Range(
                    start=Position(line=16, character=8),
                    end=Position(line=16, character=8),
                ),
                message="variable 'result' is used uninitialized whenever 'if' condition is false",
                severity=DiagnosticSeverity.Warning,
            ),
        ],

        # Only detected by CTU analysis.
        'src/divide.c': [
            Diagnostic(
                range=Range(
                    start=Position(line=4, character=21),
                    end=Position(line=4, character=21),
                ),
                message="Division by zero",
                severity=DiagnosticSeverity.Error,
            ),
        ],
    }


def expected_workspace_diagnostics(tmp_project: Path):
    return [
        WorkspaceFullDocumentDiagnosticReport(
            uri=(tmp_project / 'src' / 'main.c').as_uri(),
            items=expected_diagnostics()['src/main.c']
        ),
        WorkspaceFullDocumentDiagnosticReport(
            uri=(tmp_project / 'src' / 'divide.c').as_uri(),
            items=expected_diagnostics()['src/divide.c']
        )
    ]


# --------
# Fixtures
# --------

@pytest.fixture
def make_tmp_project(tmp_path_factory):

    def _tmp_project(project_src: Path) -> Path:
        """
        Copy the project source files at `project_src` to a new
        temporary directory, create a compilation database for
        CodeChecker there, and return the path on success.
        """
        # Adding a subfolder so that `copytree` won't error.
        project = tmp_path_factory.mktemp(project_src.name) / 'project'

        shutil.copytree(SOURCES_ROOT / project_src, project)

        proc = subprocess.run(
            ['CodeChecker', 'log', '--build', '"make"',
             '-o', './compile_commands.json'],
            cwd=project
        )
        if proc.returncode != 0:
            pytest.fail(f"Creating compilation database in {project}")

        return project

    return _tmp_project


@pytest_lsp.fixture(
    # params=['neovim', 'visual_studio_code'],
    config=ClientServerConfig(
        server_command=[sys.executable, str(CODECHECKER_LS_PATH)]
    )
)
async def client(lsp_client: LanguageClient, make_tmp_project):
    """
    Create a fixture named `client` that yields a LanguageClient
    instance already connected to a server running in a background
    process.
    """
    tmp_project: Path = make_tmp_project(project_src=Path('minimal_ctu'))
    # XXX: Monkey patching LanguageClient seems the least awkward way
    # among the awkward ways to provide test functions acces to the root
    # path.
    lsp_client.tmp_project = tmp_project  # type: ignore

    params = InitializeParams(
        capabilities=ClientCapabilities(),
        # Enabling all server features we want to test.
        initialization_options = {
            'analyze_args': ['--ctu'],
            'pull_diagnostics': True,
            'publish_diagnostics': True,
        },
        root_uri=tmp_project.as_uri(),
    )

    await lsp_client.initialize_session(params)
    yield
    await lsp_client.shutdown_session()


# -----
# Tests
# -----

@pytest.mark.parametrize(
    ('methodname', 'params_classname'),
    [('text_document_did_open', 'DidOpenTextDocumentParams'),
     ('text_document_did_save', 'DidSaveTextDocumentParams')]
)
@pytest.mark.asyncio
async def test_lsp_publish_diagnostics(
    methodname: str,
    params_classname: str,
    client: LanguageClient,
):
    """
    Puslish correct diagnostics after `textDocument/didOpen` and
    `textDocument/didSave` notifications.
    """
    tmp_project: Path = client.tmp_project  # type: ignore
    path_to_main = (tmp_project / 'src' / 'main.c')
    uri = path_to_main.as_uri()
    text = path_to_main.read_text()
    text_document = TextDocumentItem(uri=uri,
                                     language_id="c",
                                     version=1,
                                     text=text)

    # For `textDocument/didOpen`, `textDocument/didSave`...
    getattr(client, methodname)(
        getattr(lsp_types, params_classname)(
            text_document=text_document
        )
    )
    await client.wait_for_notification(TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS)

    assert uri in client.diagnostics
    result = client.diagnostics[uri]
    expected = expected_diagnostics()['src/main.c']
    assert have_same_diagnostics(expected, result)


@pytest.mark.asyncio
async def test_lsp_text_document_diagnostic(client: LanguageClient):
    """Respond to `textDocument/diagnostic` correctly."""
    tmp_project: Path = client.tmp_project  # type: ignore

    result = await client.text_document_diagnostic_async(
        params=DocumentDiagnosticParams(
            text_document=TextDocumentIdentifier(
                uri=(tmp_project / 'src' / 'main.c').as_uri()
            ),
        )
    )

    expected = RelatedFullDocumentDiagnosticReport(
        items=expected_diagnostics()['src/main.c']
    )
    assert have_same_diagnostics(expected.items, result.items)  # type: ignore


@pytest.mark.asyncio
async def test_lsp_workspace_diagnostic(client: LanguageClient):
    """Respond to `workspace/diagnostic` correctly."""
    tmp_project = client.tmp_project  # type: ignore

    result = await client.workspace_diagnostic_async(
        params=WorkspaceDiagnosticParams(previous_result_ids=[])
    )
    result = list(result.items)

    expected = expected_workspace_diagnostics(tmp_project)
    # Order should not matter.
    expected.sort(key=lambda doc_report: doc_report.uri)
    result.sort(key=lambda doc_report: doc_report.uri)

    assert len(expected) == len(result)
    for exp, res in zip(expected, result):
        assert have_same_diagnostics(exp.items, res.items)  # type: ignore


@pytest.mark.asyncio
async def test_lsp_codechecker_diagnostic(client: LanguageClient):
    """Respond to `codechecker/diagnostic` correctly."""
    tmp_project = client.tmp_project  # type: ignore

    result = await client.protocol.send_request_async(
            method=CODECHECKER_DIAGNOSTIC,
            params = {
                'previous_result_ids': [],
                'analyze_args': []
            }
    )
    result = list(result.items)

    expected = expected_workspace_diagnostics(tmp_project)
    # Order should not matter.
    expected.sort(key=lambda doc_report: doc_report.uri)
    result.sort(key=lambda doc_report: doc_report.uri)

    assert len(expected) == len(result)
    for exp, res in zip(expected, result):
        assert have_same_diagnostics(exp.items, res.items)  # type: ignore


@pytest.mark.asyncio
async def test_lsp_progress(client: LanguageClient):
    """Report progress correctly."""
    tmp_project = client.tmp_project  # type: ignore

    _ = await client.text_document_diagnostic_async(
        params=DocumentDiagnosticParams(
            text_document=TextDocumentIdentifier(
                uri=(tmp_project / 'src' / 'main.c').as_uri()
            ),
        )
    )

    progress = list(client.progress_reports.values())[0]
    assert (isinstance(progress[0], WorkDoneProgressBegin)
            and progress[0].title == 'CodeChecker analyze')
    for report in progress[1:-1]:
        assert (isinstance(report, WorkDoneProgressReport)
                and report.message
                and report.message.startswith('[INFO '))
    assert isinstance(progress[-1], WorkDoneProgressEnd)

