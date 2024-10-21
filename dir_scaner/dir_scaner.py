from pathlib import Path
import time
import os
from typing import Tuple

from directory_tree import DisplayTree


class DirScaner:
    def __init__(self, config: dict):
        self.config = config

    def scan_dir(self, dir_path: Path) -> Tuple[dict, dict]:
        result = self.scan_dir_for_documents(dir_path)
        raport = self.create_raport(result, dir_path)
        return result, raport

    def scan_dir_for_documents(self, dir_path: Path) -> dict:
        """
        Scan a directory for documents and retrieve their metadata.
        Args:
            dir_path (Path): The path to the directory to scan.

        Returns:
            dict: A dictionary containing the file paths as keys and their metadata as values.
        """
        files_paths_with_metadata = {}
        files_paths = [
            str(path)
            for path in dir_path.rglob("*")
            if path.is_file()
            and path.suffix in self.config["suffixes_of_documents_to_embedd"]
        ]
        for file_path in files_paths:
            metadata = self.get_file_metadata(file_path)
            metadata["tokens_count"] = self.count_tokens_in_file(file_path)
            files_paths_with_metadata[file_path] = metadata
        return files_paths_with_metadata

    def scan_dir_for_images(self, dir_path: Path) -> dict:
        pass

    def scan_dir_for_audio(self, dir_path: Path) -> dict:
        pass

    def create_raport(self, result: dict, dir_path: Path) -> dict:
        """
        Creates a report based on the given result.
        Args:
            result (dict): A dictionary containing metadata for each file.
        Returns:
            dict: A dictionary with the total number of files and total number of tokens.
        """
        total_files = len(result)
        total_tokens = sum(metadata["tokens_count"] for metadata in result.values())
        dir_structure = DisplayTree(dir_path, stringRep=True)
        return {
            "total_files": total_files,
            "total_tokens": total_tokens,
            "dir_structure": dir_structure,
        }

    @staticmethod
    def get_file_metadata(file_path: Path) -> dict:
        """
        Retrieves the metadata of a file.
        Args:
            file_path (str): The path to the file.
        Returns:
            dict: A dictionary containing the file's metadata, including the file path and last modified time.
        """
        file_info = os.stat(file_path)
        metadata = {
            "file_path": str(file_path),
            "last_modified": time.ctime(file_info.st_mtime),
            "file_suffix": file_path.suffix,
        }
        return metadata

    @staticmethod
    def count_tokens_in_file(file_path: str) -> int:
        """
        Counts the number of tokens in a file.
        Args:
            file_path (str): The path to the file.
        Returns:
            int: The number of tokens in the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            tokens = content.split()
            return len(tokens)
