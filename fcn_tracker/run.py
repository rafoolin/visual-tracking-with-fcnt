import argparse
import torch

from entities.particle_filter_params import ParticleFilterParams
from entities.tracker_params import TrackerParams, SelCNNParams, SNetParams, GNetParams
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
    tracker_cfg = config["tracker"]
    selcnn_cfg = tracker_cfg["selCNN"]
    snet_cfg = tracker_cfg["snet"]
    gnet_cfg = tracker_cfg["gnet"]

    # Build tracker_param
    tracker_param = TrackerParams(
        seq_path=seq_path,
        init_bbox=init_bbox,
        in_channels=tracker_cfg["in_channels"],
        pf_param=pf_param,
        roi_size=tracker_cfg["roi_size"],
        max_iter=tracker_cfg["max_iter"],
        max_iter_select=tracker_cfg["max_iter_select"],
        selcnn_param=SelCNNParams(
            bias_init=selcnn_cfg["bias_init"],
            dropout_rate=selcnn_cfg["dropout_rate"],
            in_channels=selcnn_cfg["in_channels"],
            input_size=selcnn_cfg["input_size"],
            kernel_size=selcnn_cfg["kernel_size"],
            learning_rate=selcnn_cfg["learning_rate"],
            out_channels=selcnn_cfg["out_channels"],
            padding=selcnn_cfg["padding"],
            top_k_features=selcnn_cfg["top_k_features"],
            weight_decay=selcnn_cfg["weight_decay"],
            weight_std=selcnn_cfg["weight_std"],
        ),
        snet_param=SNetParams(
            learning_rate=snet_cfg["learning_rate"],
            weight_std=snet_cfg["weight_std"],
        ),
        gnet_param=GNetParams(
            learning_rate=gnet_cfg["learning_rate"],
            weight_std=gnet_cfg["weight_std"],
        ),
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
    # TODO: use logger
    print(f"Using device: {device}")
    parser = argparse.ArgumentParser(description="Run FCNT Tracker")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    # TODO: Log config
    # Config
    config = load_config(config_path=args.config)

    # Params
    tracker_params = init_tracker_params(config)

    # Tracker
    fcnt = FCNTracker(
        params=tracker_params,
        device=device,
    )

    fcnt.initialize()
    fcnt.track()


if __name__ == "__main__":
    main()
