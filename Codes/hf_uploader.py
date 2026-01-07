# upload_to_hf.py
import os
import time
import hashlib
import json
import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

def upload_to_hf(root_path: str = 'results', include_keyword: str | None = None, case_sensitive: bool = False):
    """
    Upload files under {root_path}/Results to their respective Hugging Face repos.
    Optionally filter files to only those whose path contains `include_keyword`.

    Args:
        root_path: Base path where Results folder resides.
        include_keyword: If provided, only files whose full path contains this keyword are uploaded.
        case_sensitive: Whether keyword matching is case-sensitive.
    """
    try:
        LOCAL_FOLDER_PATH = f'{root_path}/Results'
        HF_TOKEN = "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"
        MANIFEST_FILENAME = ".upload_manifest.json"
        DELAY_BETWEEN_UPLOADS = 1.5  # seconds

        api = HfApi(token=HF_TOKEN)

        def is_checkpoint_path(path: Path) -> bool:
            return any(part.lower().startswith("checkpoint") for part in path.parts)

        def determine_repo_from_dataset(path: Path) -> str | None:
            parts_lower = [p.lower() for p in path.parts]
            if any(p.startswith(("cifar10", "cifar100", "pathmnist", "svhn")) for p in parts_lower):
                return "SolvingCO/CatastrophicOverfitting"
            elif any(p.startswith(("tissuemnist", "tinyimagenet", "cinic10", "imagenet100")) for p in parts_lower):
                return "SolvingCO2/CatastrophicOverfitting"
            else:
                return None  # dataset not recognized

        def file_sha1(path: Path) -> str:
            h = hashlib.sha1()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        def path_contains_keyword(path: Path) -> bool:
            if not include_keyword:
                return True
            haystack = str(path)
            needle = include_keyword if case_sensitive else include_keyword.lower()
            haystack_cmp = haystack if case_sensitive else haystack.lower()
            return needle in haystack_cmp

        local_path = Path(LOCAL_FOLDER_PATH)
        if not local_path.exists():
            print(f"[ERROR] Local path not found: {local_path}")
            return

        if include_keyword:
            print(f"Filtering uploads to files whose path contains: '{include_keyword}' "
                  f"(case_sensitive={case_sensitive})")

        # Group files by target repo
        repo_files_map: dict[str, list[Path]] = {}
        for file_path in local_path.rglob("*"):
            if not file_path.is_file():
                continue
            if is_checkpoint_path(file_path):
                # Skip any path components that start with 'checkpoint'
                continue
            if not path_contains_keyword(file_path):
                # Skip files that don't match the keyword filter
                # print(f"[FILTER SKIP] {file_path}")  # Uncomment for debug
                continue

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
            remote_manifest: dict[str, str] = {}
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
            updates_needed: dict[str, tuple[Path, str]] = {}
            for file_path in files:
                rel_path = str(file_path.relative_to(local_path))
                file_hash = file_sha1(file_path)
                if remote_manifest.get(rel_path) != file_hash:
                    updates_needed[rel_path] = (file_path, file_hash)

            print(f"{len(updates_needed)} files will be uploaded to {repo_id}.")

            # Upload files
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

def parse_args():
    parser = argparse.ArgumentParser(description="Upload results to Hugging Face with optional keyword filtering.")
    parser.add_argument("--root_path", type=str, default="results", help="Base path containing the Results folder.")
    parser.add_argument("--include_keyword", type=str, default=None, help="Only upload files whose path contains this keyword (e.g., 'SORA').")
    parser.add_argument("--case_sensitive", action="store_true", help="Use case-sensitive matching for the keyword.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    upload_to_hf(root_path=args.root_path, include_keyword=args.include_keyword, case_sensitive=args.case_sensitive)
