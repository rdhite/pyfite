# searching.py
# Copyright (c) 2020 Applied Research Associates, Inc.
# SPDX-License-Identifier: https://spdx.org/licenses/MIT.html

"""Utility classes for searching directories and archives.

Note:
    Though documentation may say "match" when referring to searching for patterns,
    the method ``re.match`` is not used in this module. Instead, ``re.search`` is
    used, and any finer control of searching/matching logic is expected to fall on
    the patterns provided to ``Searcher`` classes.
"""
import os
import re
import zipfile
from abc import ABC, abstractmethod
from typing import Generator, Iterable, List, Pattern

def _findAllByExtensions(self, extensions: Iterable[str], caseSensitive=False) -> List[str]:
    """Finds all files with one of the given ``extensions``.

    All extensions will be prefixed with a . before searching if one is not already present.

    Args:
        extensions (Iterable[str]): The list of extensions to search for
        caseSensitive (bool): Whether extensions should be treated as case sensitive

    Returns:
        List[str]: List of paths found with one of the given ``extensions``
    """
    extensions = [ext[1:] if ext.startswith('.') else ext for ext in extensions if len(ext) > 0]
    patternStr = f'.*\\.(?:{"|".join(extensions)})$'
    pattern = re.compile(patternStr) if caseSensitive else re.compile(patternStr, re.IGNORECASE)
    return self.findAll(pattern)

class Searcher(ABC):
    """Interface for Searchers
    """
    @abstractmethod
    def _findAll(self, pattern: Pattern[str]) -> Generator[str, None, None]:
        """Finds all files in the archive that match ``pattern``.

        Note:
            Paths starting from roots of searchers will not include leading slashes, so
            any patterns intended to match at the starts of paths (using "^") should not
            expect to match a leading slash. As such, values returned will not have leading
            slashes.

        Args:
            pattern (Pattern[str]): The pattern to use for matching

        Returns:
            Generator[str]: A generator that yields matched paths
        """
        raise NotImplementedError()

    def findAll(self, pattern: Pattern[str]) -> List[str]:
        """Finds all files in the archive that match ``pattern``.

        Note:
            Paths starting from roots of searchers will not include leading slashes, so
            any patterns intended to match at the starts of paths (using "^") should not
            expect to match a leading slash. As such, values returned will not have leading
            slashes.

        Args:
            pattern (Pattern[str]): The pattern to use for matching

        Returns:
            List[str]: The list of all files that matched ``pattern``
        """
        return list(self._findAll(pattern))

    def findFirst(self, pattern: Pattern[str]) -> str:
        """Finds the first file in the archive that matches ``pattern``.

        Note:
            Paths starting from roots of searchers will not include leading slashes, so
            any patterns intended to match at the starts of paths (using "^") should not
            expect to match a leading slash. As such, values returned will not have leading
            slashes.

        Args:
            pattern (Pattern[str]): The pattern to use for matching

        Returns:
            None: If no files matched Pattern[str]
            str: The first file that matched ``pattern``
        """
        return next(self._findAll(pattern), None)

    def findAllByExtensions(self, extensions: Iterable[str], caseSensitive=False) -> List[str]:
        """Finds all files with one of the given ``extensions``

        All extensions will be prefixed with a . before searching if one is not already present.

        Args:
            extensions (Iterable[str]): The list of extensions to search for
            caseSensitive (bool): Whether extensions should be treated as case sensitive

        Returns:
            List[str]: List of paths found with one of the given ``extensions``
        """
        extensions = [ext[1:] if ext.startswith('.') else ext for ext in extensions if len(ext) > 0]
        patternStr = f'.*\\.(?:{"|".join(extensions)})$'
        pattern = re.compile(patternStr) if caseSensitive else re.compile(patternStr, re.IGNORECASE)
        return self.findAll(pattern)

class ArchiveSearcher(Searcher):
    """Searches through archive contents without extracting any data.

    Note:
        It is assumed that the archive located at ``path`` exists
        and is of the ZIP format, but the file extension need not
        be ".zip".

        Values returned are, unless otherwise stated, full paths
        to the relevant file relative to the archive root.

    Args:
        path (str): The path to an archive to search

    Raises:
        zipfile.BadZipFile: If ``path`` isn't a ZIP file (regardless of extension)
    """
    def __init__(self, path: str):
        self._archive = zipfile.ZipFile(path, 'r')

    def __del__(self):
        self._archive.close()

    def _findAll(self, pattern: Pattern[str]) -> Generator[str, None, None]:
        """See ``Searcher._findAll``
        """
        return (entry.filename for entry in self._archive.infolist() if re.search(pattern, entry.filename) and entry.file_size > 0)

    def extractFiles(self, dest: str, files: List[str]) -> None:
        """Extracts the ``files`` in the archive to ``dest``

        The paths in ``files`` are expected to be relative to the archive root and
        directory structure within the archive will be maintained in ``dest``.

        Note:
            Paths should not be prefixed with a leading slash. If one is present,
            it will be removed.

        Args:
            dest (str): The folder into which to extract files
            files (List[str]): The list of paths within the archive to extract
        """
        for path in files:
            if path[0] in ['/', '\\']:
                path = path[1:]  # Safe to do even on empty strings
            with self._archive.open(path) as content:
                target = os.path.join(dest, path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, 'wb+') as file:
                    file.write(content.read())

class DirectorySearcher(Searcher):
    """Searches recursively through directories for files matching a pattern.

    Note:
        Values returned are, unless otherwise stated, absolute paths.

    Args:
        root (str): The root path in which to search for files

    Raises:
        NotADirectoryError: If ``root`` is not an existing directory
    """
    def __init__(self, root: str):
        self._root = root
        if not os.path.isdir(self._root):
            raise NotADirectoryError()

    def _findAll(self, pattern: Pattern[str]) -> Generator[str, None, None]:
        """See ``Searcher._findAll``
        """
        def gen():
            for root, _, files in os.walk(self._root):
                for file in files:
                    path = os.path.join(root, file)[len(self._root):]
                    if path[0] in ['/', '\\']:
                        path = path[1:]  # Safe to do even on empty strings
                    if re.search(pattern, path):
                        yield os.path.join(self._root, path)
        return gen()
