from latent_mixup_bc.benchmark import dataset_url, modern_env_id, compatibility_score


def test_dataset_urls_cover_medium_replay_v2_over_https():
    url = dataset_url("walker2d-medium-replay-v2")
    assert url == "https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5"


def test_modern_environment_mapping_is_explicit():
    assert modern_env_id("hopper-medium-v2") == "Hopper-v5"
    assert modern_env_id("halfcheetah-medium-expert-v2") == "HalfCheetah-v5"


def test_compatibility_score_uses_published_d4rl_references():
    assert compatibility_score("hopper-medium-v2", -20.272305) == 0.0
    assert compatibility_score("hopper-medium-v2", 3234.3) == 100.0
