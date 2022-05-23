from pathlib import Path
import pandas as pd


def valid_image(path):
    path = Path(path)
    return path.is_file() and path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif']


class ImclusterIO:
    def __init__(self, inputs, output, max_images:int = 0):
        self.output = Path(output)
        
        # Copy image paths into a list
        self.images = []
        for path in inputs:
            path = Path(path)

            # If it is a text file, then read each line as an image
            if path.is_dir():
                self.images += [x for x in path.iterdir() if valid_image(x)]
            elif path.suffix.lower() == ".txt":
                with open(path) as f:
                    paths_in_file = [Path(line.strip()) for line in f.readlines()]
                    self.images += [x for x in paths_in_file if valid_image(x)]
            elif valid_image(path):
                self.images.append(path)
            else:
                print(f"File '{path}' does not have a valid extension.")
        
        # truncate list of images if the user sets the maximum allowed
        if max_images and len(self.images) > max_images:
            self.images = self.images[:max_images]

        self.filenames = [image.name for image in self.images]

        if output.exists():
            df = pd.read_parquet(output, engine="pyarrow")

            # TODO check that the filenames are the same as the list
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
