import os

# Load environment variables from .env file manually
def load_env_file():
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env_file()

# Test if token is loaded
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print(f"✅ HF_TOKEN loaded successfully: {hf_token[:10]}...")
else:
    print("❌ HF_TOKEN not found")

# Test HuggingFace connection
try:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    whoami = api.whoami()
    print(f"✅ Authenticated as: {whoami['name']}")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
