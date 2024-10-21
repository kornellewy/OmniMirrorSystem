from pathlib import Path

from dir_scaner.dir_scaner import DirScaner
from database.chroma_wrapper import ChromaWrapper
from utils import read_yaml_file

from dotenv import load_dotenv

load_dotenv()


def main():
    """
    Pomysl jest taki kazda siceszka i zrudlo np jira to source i potem to zapisujemy do bazy danych.
    Narazie robimy 1 dir i 1 zrudlo.
    """

    test_dir_to_scan_path = Path(__file__).parent / "local_test_files"
    scan_config = read_yaml_file(Path(__file__).parent / "config" / "scan_config.yaml")
    data_scaner = DirScaner(config=scan_config)
    result_of_scan, raport = data_scaner.scan_dir(test_dir_to_scan_path)
    from pprint import pprint

    pprint(raport)

    database_config = read_yaml_file(
        Path(__file__).parent / "config" / "database_config.yaml"
    )
    database = ChromaWrapper(database_config)
    database.add_documents(result_of_scan)


if __name__ == "__main__":
    main()
