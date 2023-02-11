import torch, logging, os
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter

from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='retrain source and target domains with labels')
    parser.add_argument('--config-file', default="", metavar='FILE', help='path to config file');
    parser.add_argument('--imgs-dir', default="datasets/DGTA_SeaDronesSee_merged/images", type=str, help='directory of images');
    parser.add_argument('--annos-dir', default="datasets/DGTA_SeaDronesSee_merged/experiments", type=str, help='directory of jsons');
    
    args = parser.parse_args()
    return args


logger = logging.getLogger("detectron2")



def do_test(cfg, model):
    model.eval();
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    inference_on_dataset(model, val_loader, evaluator);


def do_train(cfg, model, resume = False):
    
    model.train()
    
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer = optimizer, scheduler = scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume = resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter = max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )
    
    
    data_loader = build_detection_train_loader(cfg)
    
    logger.info("Starting training from iteration {} to iteration {} with saving period {}".format(start_iter, max_iter, cfg.SOLVER.CHECKPOINT_PERIOD))

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()
            
            
            loss_dict = model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (iteration % 100 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def main():
    args = parse_args();
    print("\nCommand Line Args:", args);
    print("\n")
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file);

    # register datasets
    register_coco_instances(cfg.DATASETS.TRAIN[0], {}, args.annos_dir + "/" + cfg.DATASETS.TRAIN[0] + ".json", args.imgs_dir)
    register_coco_instances(cfg.DATASETS.TEST[0], {}, args.annos_dir + "/" + cfg.DATASETS.TEST[0] + ".json", args.imgs_dir)
    

    model = build_model(cfg);
    do_train(cfg, model);

    do_test(model, cfg.DATASETS.TEST[0])

if __name__ == "__main__":
    main();
