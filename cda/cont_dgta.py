from tqdm import tqdm

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm
import os, torch

# from core_func import compute_domain_gap_gtav, train, test
from domain_gap import compute_avg_gap, compute_dss, project_datasets

from pyJoules.energy_meter import EnergyContext
from pyJoules.handler.csv_handler import CSVHandler


# list of conditions
SOURCE_CONDITION = "CLEAR_9-15"
TARGET_CONDITIONS = ["0-1", \
                     "2-3", \
                     "4-5", \
                     "6-7", \
                     "8-9", \
                     "10-11", \
                     "12-13", \
                     "14-15", \
                     "16-17", \
                     "18-19", \
                     "20-21", \
                     "22-23"];


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='domain adaptation')
    parser.add_argument('--config-file', default="", metavar='FILE', help='path to config file');
    parser.add_argument('--projs-dir', default="", type=str, help="directory of projections")
    parser.add_argument('--gap-thr', default=0.02, type=float, help='domain gap threshold');
    parser.add_argument('--dge', action="store_true", help='do domain gap evaluation');
    parser.add_argument('--imgs-dir', default="datasets/DGTA_SeaDronesSee_merged/images", type=str, help='directory of images');
    parser.add_argument('--annos-dir', default="datasets/DGTA_SeaDronesSee_merged/experiments", type=str, help='directory of jsons');
    args = parser.parse_args()
    return args

def register_datasets(args):
    register_coco_instances(SOURCE_CONDITION + "_train", {}, args.annos_dir + "/" + "train_" + SOURCE_CONDITION + ".json", args.imgs_dir);
    register_coco_instances(SOURCE_CONDITION + "_val", {}, args.annos_dir + "/" + "val_" + SOURCE_CONDITION + ".json", args.imgs_dir);

    for cdt in TARGET_CONDITIONS:
        register_coco_instances("OVERCAST_" + cdt + "_train", {}, args.annos_dir + "/" + "train_OVERCAST_" + cdt + ".json", args.imgs_dir);
        register_coco_instances("OVERCAST_" + cdt + "_val", {}, args.annos_dir + "/" + "val_OVERCAST_" + cdt + ".json", args.imgs_dir);

def compute_domain_gap(cfg, projections_dir, current_data, new_data, n_samples = 1000):
    print("=============> COMPUTE DOMAIN GAP");
    print("Current data:")
    print(current_data);
    print("\nNew data:")
    print(new_data);

    # load model
    model = build_model(cfg);
    checkpointer = DetectionCheckpointer(model)
    print("Load model from:")
    print("\t%s" % cfg.MODEL.WEIGHTS);
    checkpointer.load(cfg.MODEL.WEIGHTS);

    # project features
    projected_current_data, projected_new_data = project_datasets(cfg, model, projections_dir, current_data, new_data, n_samples);

    # compute domain gap
    gap = compute_avg_gap(projected_current_data, projected_new_data, compute_dss, device="cuda");
    return gap


def train(cfg, cfg_new_data):
    print("=============> ADAPTING MODEL");
    print("\nCurrent training database:");
    print(cfg.DATASETS.TRAIN)

    print("\nNew training database:");
    print(cfg_new_data.DATASETS.TRAIN);

    print("\nInit model:");
    print(cfg.MODEL.WEIGHTS);


    print("\n");
    model = build_model(cfg);
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer = optimizer, scheduler = scheduler)
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume = False)
    
    start_iter = 0;
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter = max_iter);

    current_data_loader = build_detection_train_loader(cfg);
    new_data_loader = build_detection_train_loader(cfg_new_data);

    print("Starting training from iteration {} to iteration {} with saving period {}".format(start_iter, max_iter, cfg.SOLVER.CHECKPOINT_PERIOD));

    with EventStorage(start_iter) as storage:
        for current_data, new_data, iteration in tqdm(zip(current_data_loader, new_data_loader, range(start_iter, max_iter)), total=max_iter):
            iteration = iteration + 1
            storage.step()
            loss_dict = model(current_data)
            loss_dict_new = model(new_data)

            loss_dict["loss_box_reg"] += loss_dict_new["loss_box_reg"]
            loss_dict["loss_cls"] += loss_dict_new["loss_cls"];

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
            periodic_checkpointer.step(iteration)

def test(cfg, new_data_val):
    print("=============> EVALUATE");
    print("evaluation dataset:");
    print(new_data_val);
    print("\n");

    model = build_model(cfg);

    checkpointer = DetectionCheckpointer(model)
    print("Load model from:")
    print("\t%s" % cfg.MODEL.WEIGHTS);
    checkpointer.load(cfg.MODEL.WEIGHTS);


    evaluator = COCOEvaluator(new_data_val, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, new_data_val)
    res = inference_on_dataset(model, val_loader, evaluator)


def main():
    args = parse_args();
    print("\nCommand Line Args:", args);
    print("\n")

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file);

    # config file for new data
    cfg_new_data = get_cfg(); 
    cfg_new_data.INPUT.MIN_SIZE_TRAIN = cfg.INPUT.MIN_SIZE_TRAIN;
    cfg_new_data.DATALOADER.NUM_WORKERS = cfg.DATALOADER.NUM_WORKERS;
    cfg_new_data.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH

    # register datasets
    register_datasets(args);

    ######### setup directories ########
    work_dir = "work_dir/DeepGTAV"
    cfg.MODEL.WEIGHTS = work_dir + "/CLEAR_9-15/model_final.pth";
    work_dir += "/adapt_with_dge" if args.dge else "/adapt_wo_dge"
    model_dir = work_dir + "/CLEAR_9-15_and_OVERCAST"
    
    train_list = ["CLEAR_9-15_train"];

    for (i, cdt) in enumerate(TARGET_CONDITIONS):
        print("======================== ENCOUNTER NEW DOMAIN %s =====================" % ("OVERCAST_" + cdt));
        new_data_train = "OVERCAST_" + cdt + "_train";
        new_data_val = "OVERCAST_" + cdt + "_val";

        # compute domain gap if args.dge = True, else assign domain gap a large number, then the model will always be adapted
        if args.dge:
            domain_gap = compute_domain_gap(cfg=cfg, projections_dir=args.projs_dir, \
                current_data=tuple(train_list), new_data=new_data_train) 
            print("\n ==> DOMAIN GAP = %f" % domain_gap);
        else: 
            domain_gap = 999999;

        
        # if domain gap larger than a threshold, perform update detector
        if domain_gap > args.gap_thr:
            train_list.append(new_data_train);
            cfg.DATASETS.TRAIN = tuple(train_list[:-1]);
            cfg_new_data.DATASETS.TRAIN = tuple([train_list[-1]]);
            
            # create the directory to save the new model
            model_dir += "_" + cdt;
            cfg.OUTPUT_DIR = model_dir;
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

            # adapt
            csv_handler = CSVHandler(cfg.OUTPUT_DIR + '/energy.csv');
            with EnergyContext(handler=csv_handler, start_tag='train') as ctx:
                train(cfg, cfg_new_data);
            csv_handler.save_data()

            # after the training, update the current model    
            cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth" 

        
        # evaluate new detector on new target domain
        test(cfg, new_data_val);
    
    print("Done!!!");

    

if __name__ == "__main__":
    main();