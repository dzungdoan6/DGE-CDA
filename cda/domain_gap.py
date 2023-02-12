import torch, pickle, ot
import numpy as np
from torch import cdist
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.data.samplers import TrainingSampler


def do_projection(model, data_loader, projections, n):
    model.eval();
    print("Start projecting on {} batches".format(n))
    if n == -1:
        total = len(data_loader)  # inference data loader must have a fixed length
    else:
        total = n;
    
    # initialise variables to store projected features
    projected_features = {};
    for res in ["res3", "res4", "res5"]:
        projected_features[res] = np.zeros((total, projections[res].shape[0]));
        
    for inputs, idx in zip(data_loader, range(total)):
        if idx % 100 == 1:
            print("Process image %d/%d" % (idx, total));
        
        outputs, features = model(inputs)

        for res in ["res3", "res4", "res5"]:
            fea = features[res].view(len(inputs), -1) # flatten features
            proj_fea = fea.matmul(projections[res].transpose(0, 1)); # project
            projected_features[res][idx,:] = proj_fea.detach().cpu().numpy(); # store
     
    return projected_features

def load_projections(projections_dir, device = "cuda", proj_files = {"res3": "projections_n10_d16711680", "res4": "projections_n10_d8355840", "res5": "projections_n10_d4177920"}):
    
    print("\nLoad projections");
    projections = {};
    for r in ["res3", "res4", "res5"]:
        with open(projections_dir + "/" + proj_files[r], "rb") as f:
            proj = pickle.load(f);
            proj = torch.from_numpy(proj).to(device);
            projections[r] = proj;
    print("\t===> Done!!!\n")
    return projections;



# project random n_samples in current_data and new_data
def project_datasets(cfg, model, projections_dir, current_data, new_data, n_samples):
    model.eval(); # eval mode

    # load projections
    projections = load_projections(projections_dir);

    ######## project current data #########
    # count number of samples
    total_data = 0;
    for dataset_name in current_data:
        total_data += len(DatasetCatalog.get(dataset_name)) 

    
    val_loader = build_detection_test_loader(cfg, current_data, sampler = TrainingSampler(int(total_data)), batch_size=1);
    projected_current_data = do_projection(model, val_loader, projections, n_samples);

    ############# project new data ###########
    total_data = len(DatasetCatalog.get(new_data)) 
    val_loader = build_detection_test_loader(cfg, new_data, sampler = TrainingSampler(int(total_data)), batch_size=1);
    projected_new_data = do_projection(model, val_loader, projections, n_samples);

    # return
    return projected_current_data, projected_new_data;



def do_compute_mean(model, data_loader, total_data):
    model.eval(); # eval mode

    # variables to store mean features
    mean_features = {"res3": torch.zeros(16711680), 
                     "res4": torch.zeros(8355840), 
                     "res5": torch.zeros(4177920)};
    
    for inputs, idx  in zip(data_loader, range(total_data)):
        if idx % 100 == 1:
            print("\tProcess image %d/%d" % (idx, total_data));
        _, features = model(inputs);
        for res in ["res3", "res4", "res5"]:
            fea = features[res].view(-1);
            fea = fea.detach().cpu();
            mean_features[res] += fea;
        del features;
    
    for res in ["res3", "res4", "res5"]:
        mean_features[res] /= total_data;
        mean_features[res] = mean_features[res].detach().cpu().numpy() # store
    return mean_features;

def compute_mean_features(cfg, model, current_data, new_data, n_samples):

    ######## compute mean of current data #########
    # count number of samples
    total_data = 0;
    for dataset_name in current_data:
        total_data += len(DatasetCatalog.get(dataset_name)) 

    
    val_loader = build_detection_test_loader(cfg, current_data, sampler = TrainingSampler(int(total_data)), batch_size=1);
    mean_current_data = do_compute_mean(model, val_loader, n_samples);

    ############# compute mean of new data ###########
    total_data = len(DatasetCatalog.get(new_data)) 
    val_loader = build_detection_test_loader(cfg, new_data, sampler = TrainingSampler(int(total_data)), batch_size=1);
    mean_new_data = do_compute_mean(model, val_loader, n_samples);

    # return
    return mean_current_data, mean_new_data;

# compute domain gap between current_data and new_data
def compute_domain_gap(cfg, model, projections_dir, current_data, new_data, metric, n_samples = 1000):
    if metric == "DSS": 
        # project data
        projected_current_data, projected_new_data = project_datasets(cfg, model, projections_dir, current_data, new_data, n_samples);

        # compute average gap 
        gap = compute_avg_gap(projected_current_data, projected_new_data, compute_dss, device="cuda");

    if metric == "SWD":
        # project data
        projected_current_data, projected_new_data = project_datasets(cfg, model, projections_dir, current_data, new_data, n_samples);

        # compute average gap 
        gap = compute_avg_gap(projected_current_data, projected_new_data, compute_swd, device="cuda");

    if metric == "MMD":
        # compute mean
        mean_current_data, mean_new_data = compute_mean_features(cfg, model, current_data, new_data, n_samples)

        # compute average gap 
        gap = compute_avg_gap(mean_current_data, mean_new_data, compute_mmd, device="cuda");

    return gap


# compute gaps in res3, res4 and res5 and average them
def compute_avg_gap(features_1, features_2, gap_func, device="cuda"):
    gap = 0;
    for res in ["res3", "res4", "res5"]:
        fea_1 = torch.from_numpy(features_1[res]).to(device);
        fea_2 = torch.from_numpy(features_2[res]).to(device);
        gap += gap_func(fea_1, fea_2);
    return gap/3;

# compute gap on projected data then average them
def compute_dss(features_1, features_2):
    n_proj = features_1.shape[1];

    dss_val = 0;
    for i in range(n_proj):
        feas_1 = features_1[:, i];
        feas_2 = features_2[:, i];
        
        feas_1 = feas_1[:, None];
        feas_2 = feas_2[:, None];

        d = DSS(feas_1, feas_2);
        dss_val += d;
    dss_val /= n_proj;
    return dss_val;

# compute DSS
def DSS(source, target):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss


def compute_swd(features_1, features_2):
    n_proj = features_1.shape[1];

    wasserstein_distance = 0;
    for i in range(n_proj):
        feas_1 = features_1[:, i];
        feas_2 = features_2[:, i];
        
        feas_1 = feas_1[:, None];
        feas_2 = feas_2[:, None];

        cost_matrix = cdist(feas_1, feas_2)**2;
        cost_matrix = cost_matrix.detach().cpu().numpy();
        gamma = ot.emd(ot.unif(feas_1.shape[0]), ot.unif(feas_2.shape[0]), cost_matrix, numItermax=1e6);
        wasserstein_distance += np.sum(np.multiply(gamma, cost_matrix));
        
        del cost_matrix
 
    wasserstein_distance /= n_proj;
    return wasserstein_distance

def compute_mmd(features_1, features_2):
    delta = features_1 - features_2;
    return delta.dot(delta.T)