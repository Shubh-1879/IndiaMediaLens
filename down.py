#!/usr/bin/env python3
import os
import sys
from huggingface_hub import snapshot_download, __version__ as hfhub_version

# Be robust across hub versions
try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    try:
        from huggingface_hub.errors import HfHubHTTPError  # older releases
    except Exception:
        HfHubHTTPError = Exception  # last-resort fallback

def main():
    # ▶ Change: point to Gemma 7B Instruct
    repo_id = "google/gemma-7b-it"
    default_dir = "./gemma_7b_it_full"

    # Optional overrides via env vars
    local_dir = os.getenv("HF_CACHE_DIR", default_dir)
    token = os.getenv("HF_TOKEN", None)
    revision = os.getenv("HF_REVISION", "main")  # e.g., "main" or a specific tag/commit

    print(f"🔎 huggingface_hub version: {hfhub_version}")
    print(f"📥 Downloading full repo '{repo_id}' (rev: {revision}) into '{local_dir}' …")

    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # keeps a real copy in local_dir
            token=token,
            # resume_download=True is default; keeping defaults for robustness
        )
    except HfHubHTTPError as e:
        print("❌ Hugging Face download error:")
        print(e)
        print("\nTips:")
        print("• Make sure you’ve accepted the model’s license/terms on its HF page (Gemma weights are gated).")
        print("• Authenticate: `huggingface-cli login` or set HF_TOKEN env var.")
        print("• If behind a proxy/firewall, configure HTTPS proxy env vars.")
        sys.exit(1)
    except Exception as e:
        print("❌ Unexpected error:", e)
        sys.exit(1)

    print(f"✅ Download complete: {path}")

if __name__ == "__main__":
    main()
