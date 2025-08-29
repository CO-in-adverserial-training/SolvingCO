import os
import time
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

def upload_to_hf(root_path: str = 'results'):
    try:
        # Configuration
        LOCAL_FOLDER_PATH = f'{root_path}/Results'
        REPO_ID = "SolvingCO/CatastrophicOverfitting"
        HF_TOKEN = "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"
        MANIFEST_FILENAME = ".upload_manifest.json"
        DELAY_BETWEEN_UPLOADS = 1.5  # seconds

        def is_checkpoint_path(path: Path) -> bool:
            """True if path is inside/is a folder starting with 'checkpoint'."""
            return any(part.lower().startswith("checkpoint") for part in path.parts)

        def file_sha1(path: Path) -> str:
            """Calculate SHA‑1 hash of a file."""
            h = hashlib.sha1()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        api = HfApi(token=HF_TOKEN)

        # Ensure the repository exists
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)

        # Try downloading the manifest file from repo (contains {rel_path: sha1})
        remote_manifest = {}
        try:
            manifest_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MANIFEST_FILENAME,
                repo_type="model",
                token=HF_TOKEN
            )
            with open(manifest_path, "r") as f:
                remote_manifest = json.load(f)
            print(f"Loaded manifest from repo with {len(remote_manifest)} entries.")
        except EntryNotFoundError:
            print("No existing manifest found in repo — will create one.")
        except RepositoryNotFoundError:
            print(f"Repo {REPO_ID} not found; creating new one.")
        except Exception as e:
            print(f"Could not load manifest: {e}")

        # Collect local files excluding checkpoint folders
        updates_needed = {}
        local_path = Path(LOCAL_FOLDER_PATH)

        for file_path in local_path.rglob("*"):
            if file_path.is_file() and not is_checkpoint_path(file_path):
                rel_path = str(file_path.relative_to(local_path))
                file_hash = file_sha1(file_path)
                if remote_manifest.get(rel_path) != file_hash:
                    updates_needed[rel_path] = (file_path, file_hash)

        print(f"{len(updates_needed)} files will be uploaded/updated.")

        # Upload each changed file
        for idx, (rel_path, (local_file, file_hash)) in enumerate(updates_needed.items(), 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=rel_path,
                    repo_id=REPO_ID,
                    repo_type="model"
                )
                remote_manifest[rel_path] = file_hash
                print(f"[{idx}/{len(updates_needed)}] Uploaded: {rel_path}")
                time.sleep(DELAY_BETWEEN_UPLOADS)
            except Exception as e:
                print(f"Error uploading {rel_path}: {e}")

        # Upload the updated manifest file to the repo
        manifest_tmp = Path("/tmp") / MANIFEST_FILENAME
        with open(manifest_tmp, "w") as f:
            json.dump(remote_manifest, f, indent=2)

        api.upload_file(
            path_or_fileobj=str(manifest_tmp),
            path_in_repo=MANIFEST_FILENAME,
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("Updated manifest uploaded.")

    except Exception as e:
        print(f"Error pushing results to Hugging Face: {e}")

if __name__ == "__main__":
    upload_to_hf()
