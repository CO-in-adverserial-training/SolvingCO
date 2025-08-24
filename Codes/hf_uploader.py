import os
from pathlib import Path
from huggingface_hub import HfApi, Repository

def upload_to_hf():
    try:
        # Configuration
        LOCAL_FOLDER_PATH = "/home/frahmani/Github Code/SolvingCO/Codes/results/Results"  # Update this path 
        REPO_ID = "SolvingCO/CatastrophicOverfitting"  # Update with your repo ID
        HF_TOKEN = "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"  # Update with your token

        def is_checkpoint_path(path: Path) -> bool:
            """
            Returns True if the path is inside or is a folder starting with 'checkpoint'
            """
            for part in path.parts:
                if part.lower().startswith("checkpoint"):
                    return True
            return False

        def upload_folder_to_hf_repo(
            local_folder_path: str,
            repo_id: str,
            hf_token: str,
            repo_type: str = "model",
            private: bool = True
        ):
            """
            Upload a local folder to a Hugging Face repository, excluding checkpoint folders
            """
            api = HfApi(token=hf_token)
            
            # Create repository if it doesn't exist
            print("Initializing sending files...")
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    private=private,
                    exist_ok=True
                )
                print(f"Repository '{repo_id}' created/verified successfully")
            except Exception as e:
                print(f"Error creating repository: {e}")
                return
            
            # Collect files excluding checkpoint folders
            files_to_upload = []
            local_path = Path(local_folder_path)
            for file_path in local_path.rglob("*"):
                if file_path.is_file() and not is_checkpoint_path(file_path):
                    files_to_upload.append(str(file_path))
            
            # Upload only the filtered files
            try:
                for file_path in files_to_upload:
                    rel_path = str(Path(file_path).relative_to(local_folder_path))
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=rel_path,
                        repo_id=repo_id,
                        repo_type=repo_type
                    )
                print(f"Uploaded {len(files_to_upload)} files (excluding checkpoints) to '{repo_id}'")
            except Exception as e:
                print(f"Error uploading folder: {e}")

        def upload_with_progress(local_folder_path: str, repo_id: str, hf_token: str):
            """
            Alternative method using Repository class with progress tracking,
            excluding checkpoint folders
            """
            try:
                # Clone the repository
                repo = Repository(
                    local_dir="temp_repo_clone",
                    clone_from=repo_id,
                    use_auth_token=hf_token,
                    git_user="hf",
                    git_email="hf@huggingface.co"
                )

                # Copy only allowed files
                local_path = Path(local_folder_path)
                for item in local_path.rglob('*'):
                    if item.is_file() and not is_checkpoint_path(item):
                        relative_path = item.relative_to(local_path)
                        dest_path = Path("temp_repo_clone") / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(item, dest_path)
                        print(f"Copied: {relative_path}")
                
                # Commit and push
                repo.git_add(auto_lfs_track=True)
                repo.git_commit(f"Upload folder: {local_folder_path}")
                repo.git_push()
                print(f"Successfully uploaded folder with commit history")

                # Clean up
                import shutil
                shutil.rmtree("temp_repo_clone")

            except Exception as e:
                print(f"Error in upload_with_progress: {e}")

        # Upload with checkpoint exclusions
        upload_folder_to_hf_repo(
            local_folder_path=LOCAL_FOLDER_PATH,
            repo_id=REPO_ID,
            hf_token=HF_TOKEN,
            private=True
        )

    except Exception as e:
        print(f"Error pushing results to hugging face! {e}")

if __name__ == "__main__":
    upload_to_hf()
