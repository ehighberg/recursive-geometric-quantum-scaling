import yaml

def load_config(config_path: str = "config/evolution_config.yaml") -> dict:
    """
    Loads the evolution configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the configuration YAML file.

    Returns:
    - dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config