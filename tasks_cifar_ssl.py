import random
import torch
from torchvision import datasets, transforms


def get_ssl_transform():
    t = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])
    return t


def build_cifar_unlabeled(root="XXX", train=True, download=True):
    ds = datasets.CIFAR10(
        root=root,
        train=train,
        transform=None,
        download=download
    )
    return ds


def sample_ssl_tasks(args, dataset, device):
    task_num = args.task_num
    n_pairs = args.n_pairs

    t = get_ssl_transform()
    num_data = len(dataset)

    x_spt_list, x_qry_list = [], []

    for _ in range(task_num):
        idxs = random.sample(range(num_data), n_pairs)

        sup_views = []
        sup_anchors = []
        qry_views = []
        qry_anchors = []

        for idx in idxs:
            img, _ = dataset[idx]

            v1 = t(img)
            v2 = t(img)
            v3 = t(img)
            views = [v1, v2, v3]

            anchor_id = random.randint(0, 2)
            anchor = views[anchor_id]
            rest = [views[i] for i in range(3) if i != anchor_id]

            view1 = rest[0]
            view2 = rest[1]

            sup_views.append(view1)
            sup_anchors.append(anchor)

            qry_views.append(view2)
            qry_anchors.append(anchor)

        sup_views = torch.stack(sup_views, dim=0)      # [B, C, H, W]
        sup_anchors = torch.stack(sup_anchors, dim=0)  # [B, C, H, W]
        qry_views = torch.stack(qry_views, dim=0)
        qry_anchors = torch.stack(qry_anchors, dim=0)

        x_spt = torch.cat([sup_views, sup_anchors], dim=0)  # [2B, C, H, W]
        x_qry = torch.cat([qry_views, qry_anchors], dim=0)

        x_spt_list.append(x_spt)
        x_qry_list.append(x_qry)

    x_spt = torch.stack(x_spt_list, dim=0).to(device)  # [task_num, 2B, C, H, W]
    x_qry = torch.stack(x_qry_list, dim=0).to(device)

    return x_spt, x_qry
