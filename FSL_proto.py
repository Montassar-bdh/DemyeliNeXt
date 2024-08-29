import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from easyfsl.methods import PrototypicalNetworks
from monai.utils import set_determinism
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from arguments import parse_args
from data import (
    split_and_prepare_dataset,
    create_dataset_and_dataloaders,
    create_test_dataloader,
)
from explain import get_shap_values, plot_explanation
from model import FewShotDenseNet
from train import episodic_train, evaluate_model, train_model


random_seed = 3000
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def relu_inplace_to_False(module):
    for layer in module._modules.values():
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
        relu_inplace_to_False(layer)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# torch.backends.cudnn.benchmark = True

split_paths = {
    "train": {
        0: "DemyeliNeXt/Data/source1/dataset/train/NON_MS/",
        1: "DemyeliNeXt/Data/source1/dataset/train/MS/",
    },
    "valid": {
        0: "DemyeliNeXt/Data/source1/dataset/valid/NON_MS/",
        1: "DemyeliNeXt/Data/source1/dataset/valid/MS/",
    },
    "test": {
        0: "DemyeliNeXt/Data/source1/dataset/test/NON_MS/",
        1: "DemyeliNeXt/Data/source1/dataset/test/MS/",
    },
}


class_paths1 = {
    0: "DemyeliNeXt/Data/source1/dataset/test/NON_MS/",
    1: "DemyeliNeXt/Data/MSSEG_Dataset/preprocessed/",
}
#
class_paths2 = {
    0: "DemyeliNeXt/Data/source1/dataset/test/NON_MS/",
    1: "DemyeliNeXt/Data/source2/Sahloul_MS/test/MS/preprocessed/",
}

args = parse_args()
n_way = args.n_way  # Number of classes in a task
n_shot = args.n_shot  # Number of images per class in the support set
n_query = args.n_query  # Number of images per class in the query set
n_train_tasks = args.n_train_tasks
n_validation_tasks = args.n_validation_tasks
n_test_tasks = args.n_test_tasks
batch_size = args.batch_size
n_epochs = args.n_epoch
save_path = args.result_path
saved_best_epoch = args.saved_best_epoch

dataloaders = create_dataset_and_dataloaders(
    split_paths=split_paths,
    batch_size=batch_size,
    n_way=n_way,
    n_shot=n_shot,
    n_query=n_query,
    n_train_tasks=n_train_tasks,
    n_validation_tasks=n_validation_tasks,
    n_test_tasks=n_test_tasks,
    meta_dataset=True,
    enable_classic_training=True,
    shuffle=True,
)

dataloaders_MSSEG_source1 = create_test_dataloader(
    class_paths1,
    n_way=n_way,
    n_shot=n_shot,
    n_query=n_query,
    n_test_tasks=n_test_tasks,
    num_workers=0,
    shuffle=True,
)

dataloaders_source2_source1 = create_test_dataloader(
    class_paths2,
    n_way=n_way,
    n_shot=n_shot,
    n_query=n_query,
    n_test_tasks=n_test_tasks,
    num_workers=0,
    shuffle=True,
)

dataset_sizes = {
    "train": len(dataloaders["train"].dataset),
    "valid": len(dataloaders["valid"].dataset),
    "test": len(dataloaders["test"].dataset),
}

model = FewShotDenseNet(
    spatial_dims=3, in_channels=1, out_channels=2, dropout_prob=0.2, use_fc=False
).to(DEVICE)

model.set_use_fc(True)  # should be true for shap and false for training,

# summary(model, (1, 32, 64, 64), device=DEVICE)
# print("--------------------------------")

protonet = PrototypicalNetworks(
    backbone=model,
    use_softmax=False,
    feature_centering=None,
    feature_normalization=None,
).to(DEVICE)

learning_rate = 0.001
protonet_optimizer = optim.Adam(protonet.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
tb_writer = SummaryWriter(log_dir="./experiments/")

# for classical training
best_state, best_epoch = train_model(
    backbone=model,
    few_shot_classifier=protonet,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=n_epochs,
    dataloaders=dataloaders,
    dataset_sizes=dataset_sizes,
    device=DEVICE,
    validation_frequency=10,
    writer=tb_writer,
    metrics_path=save_path,
)

torch.save(
    best_state, f"./checkpoint/Densenet_batch_{batch_size}_epoch{best_epoch}.pth"
)

# for episodic training
best_state, best_epoch = episodic_train(
    protonet,
    n_epochs,
    dataloaders["train"],
    dataloaders["valid"],
    DEVICE,
    protonet_optimizer,
    criterion,
    scheduler,
    tb_writer,
    validation_frequency=1,
    metrics_path=save_path,
)

torch.save(
    best_state,
    f"./checkpoint/Densenet_shots_{n_shot}_train_{n_train_tasks}_epoch{best_epoch}.pth",
)
print("training is completed!")

checkpoint = torch.load(
    f"./checkpoint/Densenet_shots_{n_shot}_train_{n_train_tasks}_epoch{best_epoch}.pth"
)
protonet.load_state_dict(checkpoint)


msseg_source1 = os.path.join(save_path, "MSSEG_source1")
if not os.path.exists(msseg_source1):
    os.makedirs(msseg_source1)

for result_source_path in ["MSSEG_source1", "source2_source1"]:
    result_path = os.path.join(save_path, result_source_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

dataloader_source_dict = {
    "source2_source1": dataloaders_source2_source1,
    "MSSEG_source1": dataloaders_MSSEG_source1,
}

for key, dataloader_source in dataloader_source_dict.items():
    print("*" * 10)
    print(f"test model on {key} dataset ...")
    evaluate_model(
        protonet,
        dataloader_source,
        criterion,
        DEVICE,
        metrics_path=result_path,
        phase="test",
    )

print("*" * 10)
print("test model on MSSEG and monastir datasets ...")
evaluate_model(
    protonet,
    dataloaders_MSSEG_source1,
    criterion,
    device=DEVICE,
    epoch=0,
    metrics_path=msseg_source1,
    phase="test",
)


print("*" * 10)
print("test model on source2 and monastir datasets ...")
evaluate_model(
    protonet,
    dataloaders_source2_source1,
    criterion=criterion,
    device=DEVICE,
    epoch=0,
    metrics_path=source2_source1,
    phase="test",
)

print("testing is finished!")

proto_backbone = protonet.backbone
relu_inplace_to_False(proto_backbone)  # this is necessary for shap explanation


def explain_examples(
    dataset,
    background,
    support_image,
    support_label,
    support_viz_path,
    query_image,
    query_label,
    query_viz_path,
):
    print(f"explaining {dataset} example:")

    if support_label:
        gt_label = "GT: MS"
    else:
        gt_label = "GT: NON MS"

    shap_values = get_shap_values(proto_backbone, background, support_image, False)
    plot_explanation(
        shap_values,
        support_image,
        path=support_viz_path,
        gt_label=gt_label,
        fig_title="Backbone explanation of a test example with Deep SHAP",
    )

    if query_label:
        gt_label = "GT: MS"
    else:
        gt_label = "GT: NON MS"

    shap_values = get_shap_values(proto_backbone, background, query_image, False)
    plot_explanation(
        shap_values,
        query_image,
        path=query_viz_path,
        gt_label=gt_label,
        fig_title="Backbone explanation of a test example with Deep SHAP",
    )


# # #
# # #
x_train = torch.Tensor().to(DEVICE)
iter_loader = iter(dataloaders["train"])
for _ in range(6):
    batch = next(iter_loader)
    train_batch = torch.tensor(batch[0].array).to(DEVICE)
    x_train = torch.cat((x_train, train_batch))

print(x_train.shape)
background = x_train[:90]


def explain_dataset_examples(dataset, dataloader, background, save_path):
    (
        example_support_images,
        example_support_labels,
        example_query_images,
        example_query_labels,
        example_class_ids,
    ) = next(iter(dataloader))

    for i in [-1, 0]:
        label = "NON_MS" if i == 0 else "MS"

        support_image = (
            torch.tensor(example_support_images[i].array).unsqueeze(0).to(DEVICE)
        )
        support_label = example_support_labels[i]

        query_image = (
            torch.tensor(example_query_images[i].array).unsqueeze(0).to(DEVICE)
        )
        query_label = example_query_labels[i]

        support_viz_path = f"{save_path}/viz/support/{label}"
        if not os.path.exists(support_viz_path):
            os.makedirs(support_viz_path)

        query_viz_path = f"{save_path}/viz/query/{label}"
        if not os.path.exists(query_viz_path):
            os.makedirs(query_viz_path)

        explain_examples(
            dataset,
            background,
            support_image,
            support_label,
            support_viz_path,
            query_image,
            query_label,
            query_viz_path,
        )


# #####################################
explain_dataset_examples("source1", dataloaders["test"], background, save_path)
explain_dataset_examples("MSSEG", dataloaders_MSSEG_source1, background, msseg_source1)
explain_dataset_examples(
    "source2", dataloaders_source2_source1, background, source2_source1
)
