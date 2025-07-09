import argparse


from entities.particle_filter_params import ParticleFilterParams
from entities.tracker_params import TrackerParams
from utils.config import load_config


def init_tracker_params(config_path: str) -> TrackerParams:
    config = load_config(config_path=config_path)
    # Parse main values
    seq_path = config["dataset"]["seq_path"]
    init_bbox = config["dataset"]["initial_bbox"]

    # Build pf_param
    pf_param = ParticleFilterParams(
        init_bbox=init_bbox,
        affsig=config["pf_param"]["affsig"],
        p_sz=config["pf_param"]["p_sz"],
        p_num=config["pf_param"]["p_num"],
        mv_thr=config["pf_param"]["mv_thr"],
        up_thr=config["pf_param"]["up_thr"],
        roi_scale=config["pf_param"]["roi_scale"],
    )

    # Build tracker_param
    tracker_param = TrackerParams(
        seq_path=seq_path,
        init_bbox=init_bbox,
        in_channels=config["general"]["in_channels"],
        pf_param=pf_param,
        roi_size=config["general"]["roi_size"],
    )

    return tracker_param


def main():
    parser = argparse.ArgumentParser(description="Run FCNT Tracker")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    print(args.config)

    # Params
    tracker_params = init_tracker_params(args.config)
    print(f"Initial BBox: {tracker_params.init_bbox}")


if __name__ == "__main__":
    main()
