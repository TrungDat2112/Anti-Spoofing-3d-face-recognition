import yaml


def load_config(yaml_file):
    with open(yaml_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    print(cfg)
    
    return cfg