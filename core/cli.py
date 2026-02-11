import hydra
@hydra.main(config_path="../conf", config_name="config")
def main(cfg): print(cfg)
