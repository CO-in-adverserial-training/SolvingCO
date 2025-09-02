import os
import time
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

def upload_to_hf(root_path: str = 'results'):
    try:
        LOCAL_FOLDER_PATH = f'{root_path}/Results'
        HF_TOKEN = "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"
        MANIFEST_FILENAME = ".upload_manifest.json"
        DELAY_BETWEEN_UPLOADS = 1.5  # seconds

        api = HfApi(token=HF_TOKEN)

        def is_checkpoint_path(path: Path) -> bool:
            return any(part.lower().startswith("checkpoint") for part in path.parts)

        def determine_repo_from_dataset(path: Path) -> str:
            if any(part.lower().startswith(("cifar10", "cifar100", "pathmnist", "svhn")) for part in path.parts):
                return "SolvingCO/CatastrophicOverfitting"
            elif any(part.lower().startswith(("tissuemnist", "tinyimagenet", "cinic10")) for part in path.parts):
                return "SolvingCO2/CatastrophicOverfitting"
            else:
                return None  # dataset not recognized

        def file_sha1(path: Path) -> str:
            h = hashlib.sha1()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        # Group files by target repo
        repo_files_map = {}
        local_path = Path(LOCAL_FOLDER_PATH)

        for file_path in local_path.rglob("*"):
            if file_path.is_file() and not is_checkpoint_path(file_path):
                repo_id = determine_repo_from_dataset(file_path)
                if not repo_id:
                    print(f"[SKIP] {file_path} — dataset not recognized.")
                    continue
                repo_files_map.setdefault(repo_id, []).append(file_path)

        print(f"Found files for {len(repo_files_map)} repo(s).")

        # Process each repo separately
        for repo_id, files in repo_files_map.items():
            print(f"\n=== Processing repo: {repo_id} ===")

            # Ensure repo exists
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

            # Load remote manifest
            remote_manifest = {}
            try:
                manifest_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=MANIFEST_FILENAME,
                    repo_type="model",
                    token=HF_TOKEN
                )
                with open(manifest_path, "r") as f:
                    remote_manifest = json.load(f)
                print(f"Loaded manifest ({len(remote_manifest)} entries).")
            except EntryNotFoundError:
                print("No manifest in repo; starting fresh.")
            except RepositoryNotFoundError:
                print(f"Repo {repo_id} not found; creating new one.")
            except Exception as e:
                print(f"Could not load manifest: {e}")

            # Determine which files need updates
            updates_needed = {}
            for file_path in files:
                rel_path = str(file_path.relative_to(local_path))
                file_hash = file_sha1(file_path)
                if remote_manifest.get(rel_path) != file_hash:
                    updates_needed[rel_path] = (file_path, file_hash)

            print(f"{len(updates_needed)} files will be uploaded to {repo_id}.")

            # Upload
            for idx, (rel_path, (local_file, file_hash)) in enumerate(updates_needed.items(), 1):
                try:
                    api.upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=rel_path,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    remote_manifest[rel_path] = file_hash
                    print(f"[{idx}/{len(updates_needed)}] Uploaded: {rel_path}")
                    time.sleep(DELAY_BETWEEN_UPLOADS)
                except Exception as e:
                    print(f"Error uploading {rel_path} to {repo_id}: {e}")

            # Upload updated manifest
            manifest_tmp = Path("/tmp") / MANIFEST_FILENAME
            with open(manifest_tmp, "w") as f:
                json.dump(remote_manifest, f, indent=2)

            api.upload_file(
                path_or_fileobj=str(manifest_tmp),
                path_in_repo=MANIFEST_FILENAME,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"Updated manifest pushed to {repo_id}.")

    except Exception as e:
        print(f"Error pushing results to Hugging Face: {e}")

if __name__ == "__main__":
    upload_to_hf()
