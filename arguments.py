import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Set parameters for the script.")

    parser.add_argument(
        "--n_way", type=int, default=2, help="Number of classes in a task"
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=5,
        help="Number of images per class in the support set",
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=5,
        help="Number of images per class in the query set",
    )
    parser.add_argument(
        "--n_train_tasks", type=int, default=500, help="Number of training tasks"
    )
    parser.add_argument(
        "--n_validation_tasks", type=int, default=100, help="Number of validation tasks"
    )
    parser.add_argument(
        "--n_test_tasks", type=int, default=100, help="Number of test tasks"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--n_epoch", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--result_path", type=str, default=f"./results/", help="results path"
    )
    parser.add_argument(
        "--saved_best_epoch", type=int, default=1, help="saved best epoch number"
    )
    args = parser.parse_args()

    # Validate the path
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    return args
