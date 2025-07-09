import argparse
import torch

from entities.particle_filter_params import ParticleFilterParams
from entities.tracker_params import TrackerParams
from utils.config import load_config

from tracking.fcn_tracker import FCNTracker


def init_tracker_params(config: any) -> TrackerParams:
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
    # Select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    parser = argparse.ArgumentParser(description="Run FCNT Tracker")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    print(args.config)

    # Config
    config = load_config(config_path=args.config)

    # Params
    tracker_params = init_tracker_params(config)

    # Tracker
    fcnt = FCNTracker(
        config=config,
        params=tracker_params,
        device=device,
    )

    fcnt.initialize()


if __name__ == "__main__":
    main()
