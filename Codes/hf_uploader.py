import os
from pathlib import Path
from huggingface_hub import HfApi, Repository, upload_folder

def upload_to_hf():
    try:
        # Configuration 
        LOCAL_FOLDER_PATH = "/home/frahmani/SolvingCO/Codes/results"  # Update this path 
        REPO_ID = "SolvingCO/CatastrophicOverfitting"                 # Update with your repo ID
        HF_TOKEN = "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"            # Update with your token

        def upload_folder_to_hf_repo(
            local_folder_path: str,
            repo_id: str,
            hf_token: str,
            repo_type: str = "model",
            private: bool = True
        ):
            """
            Upload a local folder to a Hugging Face repository
            
            Args:
                local_folder_path: Path to the local folder to upload
                repo_id: Hugging Face repository ID (e.g., "username/repo-name")
                hf_token: Your Hugging Face authentication token
                repo_type: Type of repository ("model", "dataset", "space")
                private: Whether the repository is private
            """
            
            # Initialize the Hugging Face API
            api = HfApi(token=hf_token)
            
            # Create repository if it doesn't exist
            print("Initializing sending files...")
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    private=private,
                    exist_ok=True  # Don't error if repo already exists
                )
                print(f"Repository '{repo_id}' created/verified successfully")
            except Exception as e:
                print(f"Error creating repository: {e}")
                return
            
            # Upload the folder contents
            try:
                # Upload all files in the folder
                api.upload_folder(
                    folder_path=local_folder_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
                print(f"Successfully uploaded folder '{local_folder_path}' to '{repo_id}'")
                
            except Exception as e:
                print(f"Error uploading folder: {e}")

        def upload_with_progress(local_folder_path: str, repo_id: str, hf_token: str):
            """
            Alternative method using Repository class with progress tracking
            """
            try:
                # Clone the repository (or create if it doesn't exist)
                repo = Repository(
                    local_dir="temp_repo_clone",
                    clone_from=repo_id,
                    use_auth_token=hf_token,
                    git_user="hf",
                    git_email="hf@huggingface.co"
                )
                
                # Copy files from local folder to the cloned repository
                local_path = Path(local_folder_path)
                for item in local_path.rglob('*'):
                    if item.is_file():
                        # Maintain relative path structure
                        relative_path = item.relative_to(local_path)
                        dest_path = Path("temp_repo_clone") / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        import shutil
                        shutil.copy2(item, dest_path)
                        print(f"Copied: {relative_path}")
                
                # Commit and push changes
                repo.git_add(auto_lfs_track=True)
                repo.git_commit(f"Upload folder: {local_folder_path}")
                repo.git_push()
                
                print(f"Successfully uploaded folder with commit history")
                
                # Clean up temporary clone
                import shutil
                shutil.rmtree("temp_repo_clone")
                
            except Exception as e:
                print(f"Error in upload_with_progress: {e}")
    
        upload_folder_to_hf_repo(
            local_folder_path=LOCAL_FOLDER_PATH,
            repo_id=REPO_ID,
            hf_token=HF_TOKEN,
            private=True  # Set to False if you want a public repository
        )

    except:
        print("Error pushing results to hugging face!")

if __name__ == "__main__":
    # Method 1: Simple upload (recommended) 
    upload_to_hf()
    # Method 2: Alternative with progress tracking (uncomment to use)
    # upload_with_progress(LOCAL_FOLDER_PATH, REPO_ID, HF_TOKEN)