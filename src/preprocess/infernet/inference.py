"""

"""
from src.preprocess.infernet.common import load_infernet_model
from src.preprocess.data.selection import get_most_recent_file

if __name__ == '__main__':
    path_to_model = get_most_recent_file(
        path_to_folder="models", problem_id="problem", file_type="h5"
    )
    model = load_infernet_model(path_to_model)
