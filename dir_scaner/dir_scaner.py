from pathlib import Path
import time
import os
from typing import Tuple
from PIL import Image
from PIL.ExifTags import TAGS

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
            path
            for path in dir_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.config["suffixes_of_documents_to_embedd"]
        ]
        for file_path in files_paths:
            metadata = self.get_file_metadata(file_path)
            metadata["tokens_count"] = self.count_tokens_in_file(file_path)
            files_paths_with_metadata[file_path] = metadata
        return files_paths_with_metadata

    def scan_dir_for_images(self, dir_path: Path) -> dict:
        images_paths_with_metadata = {}
        images_paths = [
            path
            for path in dir_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.config["suffixes_of_images_to_embedd"]
        ]
        for image_path in images_paths:
            metadata = self.get_image_metadata(str(image_path))
            # TODO: Need to add image emebing to count tokens use, no host model avalable.

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

    @staticmethod
    def get_image_metadata(image_path: str) -> dict:
        """
        Extracts metadata from an image file.

        :param image_path: Path to the image file.
        :return: Dictionary containing image metadata.
        """
        metadata = {}

        try:
            # Open the image file
            with Image.open(image_path) as img:
                # Basic metadata
                metadata["Format"] = img.format
                metadata["Mode"] = img.mode
                metadata["Size"] = img.size
                metadata["Info"] = img.info

                # Extract EXIF data if available
                exif_data = img._getexif()
                if exif_data:
                    exif = {}
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        exif[tag_name] = value
                    metadata["EXIF"] = exif
                else:
                    metadata["EXIF"] = None

        except Exception as e:
            metadata["Error"] = str(e)

        return metadata
