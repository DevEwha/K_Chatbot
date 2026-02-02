from huggingface_hub import HfApi

folder_path = "/acpl-ssd30/7b_results"
repo_id = f"rinarina0429/{folder_path.split('/')[-1]}"

api = HfApi()

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model"
)
