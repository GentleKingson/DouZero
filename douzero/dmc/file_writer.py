# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import datetime
import csv
import json
import logging
import os
import threading
import time
from typing import Dict

# NOTE: ``git`` (GitPython) is intentionally NOT imported at module load.
# It is only needed by ``gather_metadata`` to stamp run metadata, and importing
# it here would force every ``import douzero.dmc`` / ``douzero.dmc.models`` and
# every ``train.py --help`` to require GitPython -- even when no git metadata is
# requested. The import now lives inside ``gather_metadata`` (lazy), so plain
# imports and ``--help`` work without GitPython installed.
#
# Git metadata is best-effort: a missing GitPython, a missing ``git`` binary,
# or a non-repo working directory degrades the recorded metadata (commit/branch
# become None with an error_type tag) but does NOT block training. The training
# loop itself never depends on git; only the run log does.


def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    # gathering git metadata
    # Lazy import: only run-logging needs GitPython. We catch the EXPECTED
    # failure modes specifically (not a bare `except Exception`) so a real bug
    # (typo, encoding error) is not silently swallowed.
    try:
        import git
    except ImportError:
        # GitPython not installed at all.
        git_data = dict(commit=None, error_type="GitPythonNotInstalled")
    else:
        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = dict(commit=None, error_type="InvalidGitRepositoryError")
        except git.GitCommandNotFound as exc:
            # GitPython imported successfully, but the ``git`` binary is not on
            # PATH (e.g. a slim container). GitCommandNotFound inherits from
            # CommandError -> GitError -> Exception, NOT OSError, so it must be
            # caught explicitly.
            git_data = dict(commit=None, error_type=type(exc).__name__)
        except TypeError:
            # Detached HEAD has no active_branch.name.
            git_data = dict(commit=None, error_type="DetachedHead")
        except OSError as exc:
            # Low-level executable / I/O failure from the git binary.
            git_data = dict(
                commit=None,
                error_type=type(exc).__name__,
            )
    # gathering slurm metadata
    if 'SLURM_JOB_ID' in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith('SLURM')]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace('SLURM_', '').replace('SLURMD_', '').lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(self,
                 xpid: str = None,
                 xp_args: dict = None,
                 rootdir: str = '~/palaas',
                 flush_interval_seconds: float = 0.0):
        if not xpid:
            # make unique id
            xpid = '{proc}_{unixtime}'.format(
                proc=os.getpid(), unixtime=int(time.time()))
        self.xpid = xpid
        self._tick = 0
        self._flush_interval_seconds = max(0.0, float(flush_interval_seconds))
        self._reuse_logs_handle = self._flush_interval_seconds > 0.0
        self._last_flush = time.monotonic()
        self._logs_handle = None
        self._lock = threading.RLock()
        self._closed = False

        # metadata gathering
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # we need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other nasty stuff).
        self.metadata['args'] = copy.deepcopy(xp_args)
        self.metadata['xpid'] = self.xpid

        formatter = logging.Formatter('%(message)s')
        self._logger = logging.getLogger('palaas/out')

        # to stdout handler
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # to file handler
        self.basepath = os.path.join(rootdir, self.xpid)

        if not os.path.exists(self.basepath):
            self._logger.info('Creating log directory: %s', self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info('Found log directory: %s', self.basepath)

        # NOTE: remove latest because it creates errors when running on slurm 
        # multiple jobs trying to write to latest but cannot find it 
        # Add 'latest' as symlink unless it exists and is no symlink.
        # symlink = os.path.join(rootdir, 'latest')
        # if os.path.islink(symlink):
        #     os.remove(symlink)
        # if not os.path.exists(symlink):
        #     os.symlink(self.basepath, symlink)
        #     self._logger.info('Symlinked log directory: %s', symlink)

        self.paths = dict(
            msg='{base}/out.log'.format(base=self.basepath),
            logs='{base}/logs.csv'.format(base=self.basepath),
            fields='{base}/fields.csv'.format(base=self.basepath),
            meta='{base}/meta.json'.format(base=self.basepath),
        )

        self._logger.info('Saving arguments to %s', self.paths['meta'])
        if os.path.exists(self.paths['meta']):
            self._logger.warning('Path to meta file already exists. '
                                 'Not overriding meta.')
        else:
            self._save_metadata()

        self._logger.info('Saving messages to %s', self.paths['msg'])
        if os.path.exists(self.paths['msg']):
            self._logger.warning('Path to message file already exists. '
                                 'New data will be appended.')

        fhandle = logging.FileHandler(self.paths['msg'])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info('Saving logs data to %s', self.paths['logs'])
        self._logger.info('Saving logs\' fields to %s', self.paths['fields'])
        if os.path.exists(self.paths['logs']):
            self._logger.warning('Path to log file already exists. '
                                 'New data will be appended.')
            if os.path.exists(self.paths['fields']):
                with open(self.paths['fields'], 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
                    self.fieldnames = rows[0] if rows else ['_tick', '_time']
            else:
                self.fieldnames = ['_tick', '_time']
        else:
            self.fieldnames = ['_tick', '_time']

        if self._reuse_logs_handle:
            self._logs_handle = open(self.paths['logs'], 'a', newline='')

    def log(self, to_log: Dict, tick: int = None,
            verbose: bool = False) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError('Cannot log to a closed FileWriter')
            if tick is not None:
                raise NotImplementedError
            else:
                to_log['_tick'] = self._tick
                self._tick += 1
            to_log['_time'] = time.time()

            old_len = len(self.fieldnames)
            for k in to_log:
                if k not in self.fieldnames:
                    self.fieldnames.append(k)
            if old_len != len(self.fieldnames):
                with open(self.paths['fields'], 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(self.fieldnames)
                self._logger.info('Updated log fields: %s', self.fieldnames)

            if verbose:
                self._logger.info('LOG | %s', ', '.join(
                    ['{}: {}'.format(k, to_log[k]) for k in sorted(to_log)]))

            def write_row(handle):
                if to_log['_tick'] == 0:
                    handle.write('# %s\n' % ','.join(self.fieldnames))
                csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(to_log)

            if self._reuse_logs_handle:
                write_row(self._logs_handle)
                if (time.monotonic() - self._last_flush
                        >= self._flush_interval_seconds):
                    self._logs_handle.flush()
                    self._last_flush = time.monotonic()
            else:
                # Preserve the original open/write/close behavior by default.
                with open(self.paths['logs'], 'a', newline='') as handle:
                    write_row(handle)

    def close(self, successful: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            if self._logs_handle is not None:
                self._logs_handle.flush()
                self._logs_handle.close()
                self._logs_handle = None
            self.metadata['date_end'] = datetime.datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S.%f')
            self.metadata['successful'] = successful
            self._save_metadata()
            self._closed = True

    def _save_metadata(self) -> None:
        with open(self.paths['meta'], 'w') as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)
