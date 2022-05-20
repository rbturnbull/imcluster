from pathlib import Path
import pandas as pd


class ImclusterIO:
    def __init__(self, images, output):
        self.images = images
        self.output = Path(output)
        self.filenames = [image.name for image in images]

        if output.exists():
            df = pd.read_parquet(output, engine="pyarrow")
        else:
            df = pd.Series(self.filenames, name="filenames").to_frame()

        self.df = df

    def has_column(self, column_name) -> bool:
        return column_name in self.df.columns

    def get_all_columns(self) -> list:
        return self.df.columns.tolist()

    def save(self):
        self.df.to_parquet(self.output, engine="pyarrow")

    def save_column(self, column_name, data, autosave=True):
        self.df[column_name] = data
        if autosave:
            self.save()

    def get_column(self, column_name) -> pd.Series:
        return self.df[column_name]
