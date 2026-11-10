import os
import zipfile
import tempfile
import pandas as pd


SUPPORTED_EXTENSIONS = [
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet",
]


def get_file_type(file_path):
    """
    Return readable file type.
    """

    extension = os.path.splitext(file_path)[1].lower()

    file_types = {
        ".csv": "CSV",
        ".xlsx": "Excel",
        ".xls": "Excel",
        ".json": "JSON",
        ".parquet": "Parquet",
        ".zip": "ZIP",
    }

    return file_types.get(extension, "Unknown")


def load_tabular_file(file_path):
    """
    Load a single tabular dataset.
    """

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(file_path)

    elif extension == ".xlsx":
        return pd.read_excel(
            file_path,
            engine="openpyxl"
        )

    elif extension == ".xls":
        return pd.read_excel(
            file_path,
            engine="xlrd"
        )

    elif extension == ".json":
        return pd.read_json(file_path)

    elif extension == ".parquet":
        return pd.read_parquet(file_path)

    else:
        raise ValueError(
            f"Unsupported dataset format: {extension}"
        )


def find_dataset_files(folder):
    """
    Find supported dataset files inside a folder.
    """

    dataset_files = []

    for root, dirs, files in os.walk(folder):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in SUPPORTED_EXTENSIONS:

                dataset_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    return dataset_files


def load_dataset(file_path):
    """
    Load normal dataset or ZIP dataset.
    """

    extension = os.path.splitext(
        file_path
    )[1].lower()

    # Normal dataset
    if extension in SUPPORTED_EXTENSIONS:

        return load_tabular_file(
            file_path
        )

    # ZIP dataset
    if extension == ".zip":

        temp_dir = tempfile.mkdtemp(
            prefix="ml_doctor_"
        )

        try:

            with zipfile.ZipFile(
                file_path,
                "r"
            ) as zip_ref:

                # Security check
                for member in zip_ref.namelist():

                    extracted_path = os.path.abspath(
                        os.path.join(
                            temp_dir,
                            member
                        )
                    )

                    if not extracted_path.startswith(
                        os.path.abspath(temp_dir)
                    ):
                        raise ValueError(
                            "Unsafe ZIP file detected."
                        )

                zip_ref.extractall(
                    temp_dir
                )

            dataset_files = find_dataset_files(
                temp_dir
            )

            if not dataset_files:

                raise ValueError(
                    "ZIP file does not contain a "
                    "supported tabular dataset."
                )

            # If multiple datasets exist,
            # use the first supported file for now.
            selected_file = dataset_files[0]

            return load_tabular_file(
                selected_file
            )

        except zipfile.BadZipFile:

            raise ValueError(
                "The uploaded ZIP file is invalid "
                "or corrupted."
            )

    raise ValueError(
        "Unsupported file format. "
        "Supported formats: CSV, XLSX, XLS, "
        "JSON, Parquet and ZIP."
    )