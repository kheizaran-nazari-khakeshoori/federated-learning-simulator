import json
# malicious toggle from config
def load_config(path="config.json"):
    try: return json.load(open(path))
    except: return {}
def save_config(d, path="config.json"):
    json.dump(d, open(path,"w"), indent=2)
