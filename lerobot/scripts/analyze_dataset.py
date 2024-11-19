from lerobot.common.datasets.factory import make_dataset
from hydra import initialize, compose

def analyze_dataset_episodes():
    initialize(config_path="../configs")
    cfg = compose(config_name="default", overrides=["policy=diffusion", "env=pusht"])
    
    dataset = make_dataset(cfg)
    
    # Get episode boundaries
    episode_starts = dataset.episode_data_index['from']
    episode_ends = dataset.episode_data_index['to']
    
    # Calculate indices for each percentage
    num_episodes = len(episode_starts)
    percentages = {
        10: num_episodes // 10,
        25: num_episodes // 4,
        50: num_episodes // 2,
        75: num_episodes * 3 // 4
    }
    
    print(f"\nTotal frames: {dataset.num_samples}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Average frames per episode: {dataset.num_samples/dataset.num_episodes:.2f}")
    
    for pct, num_eps in percentages.items():
        frame_idx = episode_ends[num_eps - 1].item()  # -1 because tensor is 0-indexed
        print(f"\nFor {pct}% episodes ({num_eps} episodes):")
        print(f"Use frame index: {frame_idx}")
        print(f"Command: python lerobot/scripts/train.py policy=diffusion env=pusht '+dataset.split=\"train[:{frame_idx}]\"'")

if __name__ == "__main__":
    dataset = analyze_dataset_episodes()