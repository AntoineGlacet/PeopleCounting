# sharepoint.py
import os
from pathlib import Path

from decouple import AutoConfig
from office365.runtime.auth.client_credential import ClientCredential
from office365.sharepoint.client_context import ClientContext
from tqdm import tqdm


def login_sharepoint():
    # get env variables and secrets from .env file
    DOTENV_FILE_PATH = Path(__file__).parent / "../data/secret/.env"
    config = AutoConfig(search_path=DOTENV_FILE_PATH)
    # login to Sharepoint
    client_credentials = ClientCredential(
        config("client_id"), config("client_secret")
    )
    ctx = ClientContext(config("site_fileserver_url")).with_credentials(
        client_credentials
    )
    return ctx, config


def download_file(
    source_file_path: Path = Path(
        r"/sites/KansaiAirportsFileServer/Shared Documents/Other/Throughput videos/test folder/test.txt"
    ),
    download_file_path: Path = Path(__file__).parent
    / "../data/dump/test.txt",
):
    # login to sharepoint
    ctx, _ = login_sharepoint()

    # open local file for writing
    with open(download_file_path, "wb") as local_file:

        # get remote file metadata
        file = (
            ctx.web.get_file_by_server_relative_path(str(source_file_path))
            .get()
            .execute_query()
        )

        # write remote file to local file

        # dirty trick for pbar update
        global pbar

        def progress(offset):
            pbar.update(1024 * 1024)

        # end of the dirty trick

        with tqdm(
            total=int(file.properties["Length"]),
            desc="downloading {}".format(str(file.properties["Name"])),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            (
                ctx.web.get_file_by_server_relative_path(
                    str(source_file_path)
                )
                .download_session(local_file, progress)
                .execute_query()
            )
    print("[Ok] file has been downloaded: {0}".format(download_file_path))


def download_folder(
    source_folder_path: Path = Path(
        r"/sites/KansaiAirportsFileServer/Shared Documents/Other/Throughput videos/test folder"
    ),
    download_folder_path: Path = Path(__file__).parent / "../data/dump",
):
    ctx, _ = login_sharepoint()

    # retrieve file collection metadata from folder
    files = (
        ctx.web.get_folder_by_server_relative_url(str(source_folder_path))
        .files.get()
        .execute_query()
    )

    # start download process (per file)
    for file in files:
        # rebuild source_file_path and download_file_path
        source_file_path = file.properties["ServerRelativeUrl"]
        # could be re-written with pathlib instead of os
        download_file_path = os.path.join(
            str(download_folder_path),
            os.path.basename(file.properties["Name"]),
        )
        download_file(source_file_path, download_file_path)

    # print a confirmation message
    print("[ok] files has been downloaded: ")
    for file in files:
        print(file.properties["Name"])
