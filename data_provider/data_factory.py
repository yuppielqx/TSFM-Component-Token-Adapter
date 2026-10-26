"""Forecasting loaders; component-code retrieval is handled inside the model."""

from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom_S, Dataset_ETT_hour, Dataset_ETT_minute


data_dict = {
    "custom": Dataset_Custom_S,
    "ett_h": Dataset_ETT_hour,
    "ett_m": Dataset_ETT_minute,
}


def data_provider(args, flag, drop_last_test=False, train_all=False):
    if args.data not in data_dict:
        raise ValueError(f"Unknown data loader {args.data!r}; choose from {tuple(data_dict)}")
    if flag not in ("train", "val", "test"):
        raise ValueError(f"Unknown split: {flag}")
    dataset = data_dict[args.data](
        model_id=args.model_id,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=0 if args.embed != "timeF" else 1,
        freq=args.freq,
        percent=args.percent,
        max_len=args.max_len,
        train_all=train_all,
        return_feature_id=args.return_feature_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag != "test",
        num_workers=args.num_workers,
        drop_last=True if flag == "train" else drop_last_test,
    )
    return dataset, loader
