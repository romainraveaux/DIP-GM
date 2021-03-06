import torch
import torch.optim as optim
import time
from pathlib import Path

from data.data_loader_multigraph import GMDataset, get_dataloader

from utils.evaluation_metric import matching_accuracy_from_lists, f1_score, get_pos_neg_from_lists, matching_accuracy, get_pos_neg

from eval import eval_model

from SinkhornModule.gm_solver import hungarian
from utils.config import cfg

from utils.utils import update_params_from_cmdline

from tensorboardX import SummaryWriter


def train_eval_model(model, criterion, optimizer, dataloader, num_epochs, writer, resume=False, start_epoch=0):
    print("Start training...")

    since = time.time()
    dataloader["train"].dataset.set_num_graphs(cfg.TRAIN.num_graphs_in_matching_instance)
    dataset_size = len(dataloader["train"].dataset)


    device = next(model.parameters()).device
    print("model on device: {}".format(device))

    checkpoint_path = Path(cfg.model_dir) / "params"
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        params_path = os.path.join(cfg.warmstart_path, f"params.pt")
        print("Loading model parameters from {}".format(params_path))
        model.load_state_dict(torch.load(params_path))

        optim_path = os.path.join(cfg.warmstart_path, f"optim.pt")
        print("Loading optimizer state from {}".format(optim_path))
        optimizer.load_state_dict(torch.load(optim_path))

    # Evaluation only
    if cfg.evaluate_only:
        assert resume
        print(f"Evaluating without training...")
        accs, f1_scores = eval_model(model, dataloader["test"])
        acc_dict = {
            "acc_{}".format(cls): single_acc for cls, single_acc in zip(dataloader["train"].dataset.classes, accs)
        }
        f1_dict = {
            "f1_{}".format(cls): single_f1_score
            for cls, single_f1_score in zip(dataloader["train"].dataset.classes, f1_scores)
        }
        acc_dict.update(f1_dict)
        acc_dict["matching_accuracy"] = torch.mean(accs)
        acc_dict["f1_score"] = torch.mean(f1_scores)

        time_elapsed = time.time() - since
        print(
            "Evaluation complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
            )
        )
        return model, acc_dict

    lr_params = cfg.TRAIN.lr_schedule
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_params["lr_milestones"], gamma=lr_params["lr_decay"]
    )

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        print("Epoch {}/{}".format(epoch, num_epochs - 1))

        model.train()  # Set model to training mode

        print("lr = " + ", ".join(["{:.2e}".format(x["lr"]) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        running_loss = 0.0
        running_acc = 0.0
        epoch_acc = 0.0
        running_f1 = 0.0
        epoch_f1 = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for inputs in dataloader["train"]:
            data_list = [_.cuda() for _ in inputs["images"]]
            points_gt_list = [_.cuda() for _ in inputs["Ps"]]
            n_points_gt_list = [_.cuda() for _ in inputs["ns"]]
            graphs_list = [_.to("cuda") for _ in inputs["graphs"]]
            perm_mat_list = [perm_mat.cuda() for perm_mat in inputs["gt_perm_mat"]]

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                s_pred_list = model(data_list, points_gt_list, graphs_list, n_points_gt_list, perm_mat_list)

                if cfg.MODULE == "MIP":
                    loss = sum([criterion(s_pred, perm_mat) for s_pred, perm_mat in zip(s_pred_list, perm_mat_list)])
                    loss /= len(s_pred_list)
                
                    # backward + optimize
                    loss.backward()
                    optimizer.step()
                    
                    tp, fp, fn = get_pos_neg_from_lists(s_pred_list, perm_mat_list)
                    f1 = f1_score(tp, fp, fn)
                    acc, _, __ = matching_accuracy_from_lists(s_pred_list, perm_mat_list)
                elif cfg.MODULE == "Sinkhorn":
                    ns_src, ns_dst = n_points_gt_list[0], n_points_gt_list[1]
                    loss = criterion(s_pred_list, perm_mat_list[0], ns_src, ns_dst)
                    
                    # backward + optimize
                    loss.backward()
                    optimizer.step()

                    pred_perm = hungarian(s_pred_list.detach(), ns_src, ns_dst)
                    acc, _, __ = matching_accuracy(pred_perm, perm_mat_list[0])
                    tp, fp, fn = get_pos_neg(pred_perm, perm_mat_list[0])
                    f1 = f1_score(tp, fp, fn)

                # statistics
                bs = perm_mat_list[0].size(0)
                running_loss += loss.item() * bs  # multiply with batch size
                epoch_loss += loss.item() * bs
                running_acc += acc.item() * bs
                epoch_acc += acc.item() * bs
                running_f1 += f1.item() * bs
                epoch_f1 += f1.item() * bs

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * bs / (time.time() - running_since)
                    loss_avg = running_loss / cfg.STATISTIC_STEP / bs
                    acc_avg = running_acc / cfg.STATISTIC_STEP / bs
                    f1_avg = running_f1 / cfg.STATISTIC_STEP / bs
                    print(
                        "Epoch {:<4} Iter {:<4} {:>4.2f}sample/s Loss={:<8.4f} Accuracy={:<2.3} F1={:<2.3}".format(
                            epoch, iter_num, running_speed, loss_avg, acc_avg, f1_avg
                        )
                    )

                    running_acc = 0.0
                    running_f1 = 0.0
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size
        epoch_acc = epoch_acc / dataset_size
        epoch_f1 = epoch_f1 / dataset_size

        writer.add_scalars("Train", {"epoch_loss":epoch_loss, 
                                    "epoch_acc":epoch_acc,}, epoch)

        if cfg.save_checkpoint:
            base_path = Path(checkpoint_path / "{:04}".format(epoch + 1))
            Path(base_path).mkdir(parents=True, exist_ok=True)
            path = str(base_path / "params.pt")
            torch.save(model.state_dict(), path)
            torch.save(optimizer.state_dict(), str(base_path / "optim.pt"))

        print(
            "Over whole epoch {:<4} -------- Loss: {:.4f} Accuracy: {:.3f} F1: {:.3f}".format(
                epoch, epoch_loss, epoch_acc, epoch_f1
            )
        )
        print()

        # Eval in each epoch
        accs, f1_scores = eval_model(model, dataloader["test"])
        acc_dict = {
            "acc_{}".format(cls): single_acc for cls, single_acc in zip(dataloader["train"].dataset.classes, accs)
        }
        f1_dict = {
            "f1_{}".format(cls): single_f1_score
            for cls, single_f1_score in zip(dataloader["train"].dataset.classes, f1_scores)
        }
        acc_dict.update(f1_dict)
        acc_dict["matching_accuracy"] = torch.mean(accs)
        acc_dict["f1_score"] = torch.mean(f1_scores)

        writer.add_scalars("Eval", {"eval_acc":acc_dict["matching_accuracy"]}, epoch)

        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60
        )
    )

    return model, acc_dict


if __name__ == "__main__":
    from utils.dup_stdout_manager import DupStdoutFileManager

    cfg = update_params_from_cmdline(default_params=cfg)
    import json
    import os

    os.makedirs(cfg.model_dir, exist_ok=True)
    with open(os.path.join(cfg.model_dir, "settings.json"), "w") as f:
        json.dump(cfg, f)

    torch.manual_seed(cfg.RANDOM_SEED)

    dataset_len = {"train": cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, "test": cfg.EVAL.SAMPLES}

    image_dataset = {
        x: GMDataset(cfg.DATASET_NAME, sets=x, length=dataset_len[x], obj_resize=(256, 256)) for x in ("train", "test")
    }
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == "test"), shuffle=True) for x in ("train", "test")}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.MODULE == "MIP":
        from module.model import Net
        from module.loss_function import HammingLoss
        model = Net()
        criterion = HammingLoss()
    elif cfg.MODULE == "Sinkhorn":
        from SinkhornModule.model import Net
        from module.loss_function import CrossEntropyLoss
        model = Net()
        criterion = CrossEntropyLoss()


    model = model.cuda()

    backbone_params = list(model.node_layers.parameters()) + list(model.edge_layers.parameters())
    backbone_params += list(model.final_layers.parameters())

    backbone_ids = [id(item) for item in backbone_params]

    new_params = [param for param in model.parameters() if id(param) not in backbone_ids]
    opt_params = [
        dict(params=backbone_params, lr=cfg.TRAIN.LR * 0.01),
        dict(params=new_params, lr=cfg.TRAIN.LR),
    ]
    optimizer = optim.Adam(opt_params)

    if not Path(cfg.model_dir).exists():
        Path(cfg.model_dir).mkdir(parents=True)

    writer = SummaryWriter(str(Path(cfg.model_dir) / ("runs")))

    num_epochs = cfg.TRAIN.lr_schedule.num_epochs
    with DupStdoutFileManager(str(Path(cfg.model_dir) / ("train_log.log"))) as _:
        model, accs = train_eval_model(
            model,
            criterion,
            optimizer,
            dataloader,
            num_epochs=num_epochs,
            writer=writer,
            resume=cfg.warmstart_path is not None,
            start_epoch=0,
        )
