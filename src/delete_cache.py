import shutil
import pathlib

shutil.rmtree('./cache')
pathlib.Path('./cache').mkdir(parents=True, exist_ok=True)
