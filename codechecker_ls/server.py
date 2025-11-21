#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import selectors
import subprocess
import urllib.parse
import uuid
from enum import Enum
from pathlib import PosixPath
from signal import SIGTERM, SIGKILL
from typing import Any, Callable, Generator, Optional, Type, TypeAlias, Union

import attrs
from lsprotocol import types
from lsprotocol.types import (
    INITIALIZE,
    SHUTDOWN,
    Diagnostic,
    DiagnosticSeverity,
    FullDocumentDiagnosticReport,
    InitializeParams,
    Position,
    Range,
    WorkDoneProgressBegin,
    WorkDoneProgressEnd,
    WorkDoneProgressReport,
    WorkspaceDiagnosticParams,
    WorkspaceDiagnosticReport,
    WorkspaceDiagnosticRequest,
    WorkspaceFullDocumentDiagnosticReport,
)
from pygls.cli import start_server
from pygls.exceptions import (
    JsonRpcInternalError,
    JsonRpcRequestCancelled
)
from pygls.lsp.server import LanguageServer
from pygls.protocol import LanguageServerProtocol


# -------------------
# Types and constants
# -------------------

class Signal(Enum):
    LSP_CANCEL_NOTIFICATION = 'LSP_CANCEL_NOTIFICATION'
    SERVER_SHUTDOWN = 'SERVER_SHUTDOWN'

SuccStat: TypeAlias = tuple[bool, Union[int, Signal]]
Uri: TypeAlias = str

CODECHECKER_DIAGNOSTIC = 'codechecker/diagnostic'


# ------------
# Global state
# ------------

LOG = logging.getLogger(__name__)

INIT_OPTIONS = {
    'analyze_args': [],
}


# ------------------
# Protocol extension
# ------------------

@attrs.define(kw_only=True)  # `kw_only` -> https://stackoverflow.com/a/72195784
class CodeCheckerDiagnosticParams(WorkspaceDiagnosticParams):
    analyze_args: list[str] = attrs.field()
    """Additional arguments to be passed to `CodeChecker analyze`."""


@attrs.define(kw_only=True)
class CodeCheckerDiagnosticRequest(WorkspaceDiagnosticRequest):
    method: str = CODECHECKER_DIAGNOSTIC  # type: ignore
    params: CodeCheckerDiagnosticParams = attrs.field()  # type: ignore


class CodeCheckerLanguageServerProtocol(LanguageServerProtocol):
    """
    Extends the protocol with a custom request type derived from
    workspace/diagnostic.

    This allows client integrations:
    - full control over when to send requests to the server
    - passing custom arguments to `CodeChecker analyze` on a per-call
      basis in the editor
    """

    def get_message_type(self, method: str) -> Optional[Type]:  # type: ignore
        if method == CodeCheckerDiagnosticRequest.method:
            return CodeCheckerDiagnosticRequest
        return super().get_message_type(method)


# -----
# Utils
# -----

# NOTE: According to LSP, if severity is omitted, it is up to the client
# to interpret diagnostics as they like.
def to_lsp_diagnostic_severity(
    codechecker_severity: str,
) -> Optional[DiagnosticSeverity]:
    if codechecker_severity == 'HIGH':
        return DiagnosticSeverity.Error
    elif codechecker_severity == 'MEDIUM':
        return DiagnosticSeverity.Warning
    elif codechecker_severity == 'LOW':
        return DiagnosticSeverity.Information
    elif codechecker_severity == 'STYLE':
        return DiagnosticSeverity.Hint


def to_lsp_diagnostic(report) -> Diagnostic:
    return Diagnostic(
        source='CodeChecker(' + report['analyzer_name'] + ')',
        message=report['message'],
        severity=to_lsp_diagnostic_severity(report['severity']),
        range=Range(
            start=Position(
                line=report['line'] - 1, character=report['column'] - 1
            ),
            end=Position(
                line=report['line'] - 1, character=report['column'] - 1
            ),
        ),
        # Including the full original report structure (primarily to
        # allow clients to extract reproduction steps).
        data=report,
    )


# TODO: Make these configurable in `init_opts`.
def get_default_paths(workspace_folder) -> tuple[str, str]:
    codechecker_folder = os.path.join(workspace_folder, '.codechecker')
    reports_path = os.path.join(codechecker_folder, 'reports')
    json_path = os.path.join(codechecker_folder, 'reports.json')
    return reports_path, json_path


def log_parse_err(err) -> None:
    if err == 1:
        LOG.info("Parsing to JSON failed: CodeChecker error.")


def log_analyze_err(err) -> None:
    if err == Signal.LSP_CANCEL_NOTIFICATION:
        LOG.info("Analysis canceled by user.")
    elif err == Signal.SERVER_SHUTDOWN:
        LOG.info("Analysis canceled because of server shutdown.")
    else:
        LOG.info(f"Analysis failed with error code: {err}.")


# ---------
# Processes
# ---------

def run_parse(src_path, out_path) -> SuccStat:
    """Run `CodeChecker parse`, exporting to JSON."""
    args = [
        'CodeChecker', 'parse', src_path,
        '-o', out_path,
        '--export', 'json',
        '--trim-path-prefix',
    ]
    proc = subprocess.run(args, capture_output=True, timeout=60)
    return (proc.returncode != 1, proc.returncode)


def run_analyze(args) -> Generator[str, Signal, SuccStat]:
    """
    Run `CodeChecker analyze` with the given arguments, yielding its
    stdout along the way.
    """
    with subprocess.Popen(['CodeChecker', 'analyze', *args],
                          start_new_session=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,
                          text=True) as proc:
        LOG.info("`CodeChecker analyze` called")

        # Along with emitting `analyze` output for progress report
        # purposes, we would simultaneously want to poll for cancel
        # signals at some fixed interval.
        with selectors.DefaultSelector() as sel:
            sel.register(proc.stdout, selectors.EVENT_READ)  # type: ignore

            while True:
                events = sel.select(timeout=0.2)
                # An explicit check here instead of relying on EOF
                # protects us if `analyze` crashes for whatever reason.
                if proc.poll() is not None:
                    LOG.info("`CodeChecker analyze` finished")
                    return (proc.returncode == 0, proc.returncode)

                for key, _ in events:
                    line = key.fileobj.readline()  # type: ignore
                    if line:
                        LOG.debug(f"`CodeChecker analyze` stdout: {line}")
                        yield line

                signal = yield  # type: ignore
                if signal and (signal in Signal):
                    LOG.info("Killing `CodeChecker analyze`...")
                    try:
                        os.killpg(proc.pid, SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, SIGKILL)
                    except ProcessLookupError:
                        pass
                    return (False, signal)


# ------
# Server
# ------

class CodeCheckerLanguageServer(LanguageServer):
    """Language Server implementation for CodeChecker."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = asyncio.Lock()
        self.active_task: Optional[Generator[str, Signal, SuccStat]] = None

    def kill_active_task(self, signal: Signal):
        if self.active_task:
            try:
                self.active_task.send(signal)
            except TypeError:
                pass

    def shutdown(self) -> None:
        LOG.info("Shutting down codechecker-ls")
        self.kill_active_task(Signal.SERVER_SHUTDOWN)

    async def with_progress_report(
            self,
            generator: Callable[..., Generator[str, Signal, SuccStat]],
            *generator_args: ...,
            title="",
            cancellable=True) -> SuccStat:
        """
        Exhaust the provided `generator` while using the yielded output
        to report LSP work done progress along the way.
        """
        token = str(uuid.uuid4())
        await self.work_done_progress.create_async(token)
        self.work_done_progress.begin(
            token,
            WorkDoneProgressBegin(
                kind='begin', title=title, cancellable=cancellable
            ),
        )
        try:
            self.active_task = generator(*generator_args)
            while True:
                # Pygls handles work done progress cancellation via
                # Future objects registered for each token on creation;
                # when receiving `window/workDoneProgress/cancel`, it
                # calls `cancel()` on the corresponding future.
                await asyncio.sleep(0)  # allow receiving a cancellation request
                if self.work_done_progress.tokens[token].cancelled():
                    self.work_done_progress.report(
                        token,
                        WorkDoneProgressReport(
                            kind='report',
                            message="Notification `window/workDoneProgress/cancel` received",
                        ),
                    )
                    self.active_task.send(Signal.LSP_CANCEL_NOTIFICATION)
                else:
                    msg = next(self.active_task)
                    if msg:  # gen might just listen to `send()`
                        self.work_done_progress.report(
                            token,
                            WorkDoneProgressReport(
                                kind='report',
                                message=msg,
                                cancellable=cancellable
                            ),
                        )
        except StopIteration as e:
            LOG.debug(f"StopIteration: {e.value}")
            # If `self.active_task` received a kill signal from
            # `shutdown()`, it is already exhausted (no value).
            # FIXME: Decouple shutdown logic, this is a dirty workaround.
            return e.value or (False, Signal.SERVER_SHUTDOWN)
        finally:
            self.work_done_progress.end(token, WorkDoneProgressEnd(kind='end'))
            self.active_task = None

    async def run_codechecker(self, analyze_args, reports_path, json_path) -> Any:
        """
        Run `CodeChecker analyze` with the given arguments, `parse` the
        reports to JSON, and load the parsed data.
        """
        # We are using an async coroutine for cancellation capability,
        # but otherwise we want to behave like a blocking call (queueing
        # requests).
        async with self.lock:
            success, status = await self.with_progress_report(
                run_analyze, analyze_args, title='CodeChecker analyze'
            )
            if not success:
                log_analyze_err(status)
                if isinstance(status, Signal):
                    # HACK: This error will be handled silently by
                    # Neovim, so sending this on server shutdown too.
                    raise JsonRpcRequestCancelled
                else:
                    raise JsonRpcInternalError(
                            message=f'CodeChecker analyze error: {status}')

            success, status = run_parse(reports_path, json_path)
            if not success:
                log_parse_err(status)
                raise JsonRpcInternalError(
                        message=f'CodeChecker parse error: {status}')

            with open(json_path, 'r') as f:
                try:
                    parsed = json.load(f)
                    return parsed
                except json.JSONDecodeError:
                    LOG.info("Decoding " + json_path + " failed.")

    def get_workspace_folder(self) -> str:
        folders = list(self.workspace.folders.values())
        return folders[0].name if folders else '.'

    async def get_document_diagnostics(self, uri: Uri) -> list[Diagnostic]:
        """
        Run `CodeChecker analyze` on the path given as `uri`, parse the
        results into JSON, and convert it to a single LSP document
        diagnostic report object.
        """
        workspace_folder = self.get_workspace_folder()
        reports_path, json_path = get_default_paths(workspace_folder)
        input_path = urllib.parse.urlparse(uri).path

        optional_args = INIT_OPTIONS.get('analyze_args', [])
        analyze_args = [input_path, '-o', reports_path, *optional_args]

        parsed = await self.run_codechecker(analyze_args, reports_path, json_path)
        if not parsed:
            raise JsonRpcInternalError()

        return [to_lsp_diagnostic(report) for report in parsed['reports']]

    async def get_workspace_diagnostics(
        self, analyze_args: list[str] = []
    ) -> WorkspaceDiagnosticReport:
        """
        Run `CodeChecker analyze`, parse the results into JSON, and
        convert it to a single LSP workspace diagnostic report object.
        """
        workspace_folder = self.get_workspace_folder()
        reports_path, json_path = get_default_paths(workspace_folder)
        input_path = os.path.join(workspace_folder, 'compile_commands.json')

        optional_args = analyze_args or INIT_OPTIONS.get('analyze_args', [])
        # Note: For single-file analysis, `--file foo` still works.
        analyze_args = [input_path, '-o', reports_path, *optional_args]

        parsed = await self.run_codechecker(analyze_args, reports_path, json_path)
        if not parsed:
            raise JsonRpcInternalError()

        # Workspace diagnostics is expected as an array of document
        # diagnostics.

        diags: dict[Uri, list[Diagnostic]] = {}
        for report in parsed['reports']:
            uri: Uri = PosixPath(report['file']['original_path']).as_uri()
            if not diags.get(uri):
                diags[uri] = []
            diags[uri].append(to_lsp_diagnostic(report))

        doc_reports: list[WorkspaceFullDocumentDiagnosticReport] = []
        for uri, diagnostics in diags.items():
            doc = self.workspace.get_text_document(uri)
            doc_reports.append(
                WorkspaceFullDocumentDiagnosticReport(
                    uri=uri,
                    version=doc.version,
                    items=diagnostics
                )
            )
        return WorkspaceDiagnosticReport(items=doc_reports)

    async def publish_diagnostics(self, uri: Uri) -> None:
        """
        Get diagnostics for the given `uri`, and send the result in a
        `textDocument/publishDiagnostics` notification to the client.
        """
        doc = self.workspace.get_text_document(uri)
        diags = await self.get_document_diagnostics(doc.uri)
        self.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=doc.uri,
                version=doc.version,
                diagnostics=diags,
            )
        )


SERVER = CodeCheckerLanguageServer(
    name='codechecker_ls',
    version='0.0.1',
    protocol_cls=CodeCheckerLanguageServerProtocol
)


@SERVER.feature(SHUTDOWN)
def shutdown(ls: CodeCheckerLanguageServer, *args) -> None:
    """LSP handler for `shutdown` request."""
    ls.shutdown()


@SERVER.feature(INITIALIZE)
def initialize(params: InitializeParams) -> None:
    """LSP handler for `initialize` request."""
    init_opts: dict = params.initialization_options or {}

    for k, v in init_opts.items():
        INIT_OPTIONS[k] = v

    default_log_path = os.path.join(
        params.workspace_folders[0].name if params.workspace_folders else '.',
        'codechecker_ls_log.txt'
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s',
        filename=os.path.expanduser(init_opts.get('log_path', default_log_path))
    )
    LOG.info("Starting codechecker-ls")

    # Dynamic registration of capabilities based on client settings.

    # NOTES:
    # Standard requests can be problematic (lacking means to decide
    # about and skip redundant requests), because clients might trigger
    # them far too frequently, without allowing much control over the
    # behavior.
    # Cf. https://github.com/astral-sh/ruff/pull/18939#issue-3176261869

    # If provided, document diagnostics should be opt-in, using the older
    # "publish" modell, responding to `didSave` and `didOpen` notifications.
    # Note that clients can also trigger the custom `codechecker/diagnostic`
    # request on relevant events themselves, requesting `analyze` for the
    # current file only, which might be a feasible alternative.

    if init_opts.get('publish_diagnostics'):

        @SERVER.feature(types.TEXT_DOCUMENT_DID_OPEN)
        async def did_open(
            ls: CodeCheckerLanguageServer,
            params: types.DidOpenTextDocumentParams
        ) -> None:
            """LSP handler for `textDocument/didOpen` notification."""
            await ls.publish_diagnostics(params.text_document.uri)

        @SERVER.feature(types.TEXT_DOCUMENT_DID_SAVE)
        async def did_save(
            ls: CodeCheckerLanguageServer,
            params: types.DidSaveTextDocumentParams
        ) -> None:
            """LSP handler for `textDocument/didSave` notification."""
            await ls.publish_diagnostics(params.text_document.uri)

    if init_opts.get('pull_diagnostics'):

        # FIXME/RESEARCH: Without registering `textDocument/diagnostic`,
        # Neovim does not see `workspace/diagnostic` capability in the
        # server. Exposing `textDocument/diagnostic` just for this is
        # far from elegant (continuously accepting requests destined to
        # fall on deaf ears).

        @SERVER.feature(
            types.TEXT_DOCUMENT_DIAGNOSTIC,
            types.DiagnosticOptions(
                inter_file_dependencies=True,
                workspace_diagnostics=True
            ),
        )
        async def document_diagnostic(
            ls: CodeCheckerLanguageServer,
            params: types.DocumentDiagnosticParams
        ) -> FullDocumentDiagnosticReport:
            """LSP handler for `textDocument/diagnostic` request."""
            diags = await ls.get_document_diagnostics(params.text_document.uri)
            return FullDocumentDiagnosticReport(items=diags)


        @SERVER.feature(types.WORKSPACE_DIAGNOSTIC)
        async def workspace_diagnostic(
            ls: CodeCheckerLanguageServer,
            params: WorkspaceDiagnosticParams
        ) -> WorkspaceDiagnosticReport:
            """LSP handler for `workspace/diagnostic` request."""
            return await ls.get_workspace_diagnostics()

    else:
        # HACK: Registering `textDocument/diagnostic` capability is necessary
        # for Neovim to be able to use the corresponding handler function.
        # (We use that in the client integration plugin to handle
        # `codechecker/diagnostic`, because for the moment there is no exposed
        # `workspace/diagnostic` handler.)
        @SERVER.feature(
            types.TEXT_DOCUMENT_DIAGNOSTIC,
            types.DiagnosticOptions(
                inter_file_dependencies=True, workspace_diagnostics=True
            ),
        )
        def document_diagnostic_placeholder(ls, params):
            return None


@SERVER.feature(CODECHECKER_DIAGNOSTIC)
async def codechecker_diagnostic(
    ls: CodeCheckerLanguageServer,
    # FIXME/RESEARCH: `params` is still deserialized to plain Object here.
    # Cf. https://github.com/openlawlibrary/pygls/discussions/441
    params: CodeCheckerDiagnosticParams
):
    """LSP handler for `codechecker/diagnostic` request."""
    return await ls.get_workspace_diagnostics(analyze_args=params.analyze_args)


def main():
    start_server(SERVER)


if __name__ == '__main__':
    main()

