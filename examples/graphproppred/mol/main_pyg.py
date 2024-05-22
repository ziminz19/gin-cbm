import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN, GNN_CBM
import dgl
import torch
from torch_geometric.data import Data
from functools import reduce

from tqdm import tqdm
import argparse
import time
import numpy as np
import json

import matplotlib.pyplot as plt 
import io 
from PIL import Image
from pickle import dump

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            #pred = model(batch)
            pred_concept = model.gnn(batch)
            pred_label = model.concept_pred_head(pred_concept)

            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                #loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                #print(pred_concept.shape)
                #print(batch.concept.shape)
                loss_concept = reg_criterion(pred_concept.to(torch.float32), batch.concept.reshape(len(batch),20).to(torch.float32))
                loss_label = cls_criterion(pred_label.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss = loss_concept + loss_label
            else:
                #loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss_concept = reg_criterion(pred_concept.to(torch.float32), batch.concept.reshape(len(batch),20).to(torch.float32))
                loss_label = reg_criterion(pred_label.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss = loss_concept + loss_label
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def dgl_to_pyg(dgl_graph):
    # Extract node features (if available)
    node_features = dgl_graph.ndata['attr'] if 'attr' in dgl_graph.ndata else None

    # Extract edge indices
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)

    # Extract edge features (if available)
    edge_features = dgl_graph.edata['edge_attr'] if 'edge_attr' in dgl_graph.edata else None

    # Create PyG Data object
    pyg_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    return pyg_graph

def merge_pyg_graphs(graph1, graph2):
    # Concatenate node features
    if graph1.x is not None and graph2.x is not None:
        node_features = torch.cat([graph1.x, graph2.x], dim=0)
    else:
        node_features = None

    # Concatenate edge indices, adjusting the indices of the second graph
    edge_index2 = graph2.edge_index + graph1.num_nodes
    edge_index = torch.cat([graph1.edge_index, edge_index2], dim=1)

    # Concatenate edge features
    if graph1.edge_attr is not None and graph2.edge_attr is not None:
        edge_features = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=0)
    else:
        edge_features = None

    # Create a new PyG Data object with the merged features
    merged_graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

    return merged_graph

def add_attributes(dataset):
    data_list = []
    for i, data in enumerate(dataset):
        #print(torch.from_numpy(concept_value_llm[i]).shape)
        data.concept = torch.from_numpy(concept_value_llm[i])
        data_list.append(data)
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### Load our generated value
    def read_from_json(file_name: str) -> dict:
        with open(file_name, 'r') as f:
            return json.load(f)

    def fill_nones_with_column_average(X):
        column_sum = [0] * X.shape[1]
        column_n = [0] * X.shape[1]
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] != None:
                    column_sum[j] += X[i, j]
                    column_n[j] += 1
        column_mean = [column_sum[i] / column_n[i] for i in range(len(column_sum))]

        out = np.zeros(X.shape)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if X[i, j] == None:
                    out[i, j] = column_mean[j]
                else:
                    out[i, j] = X[i, j]
        return out
    
    llm_generated_raw = read_from_json("bh_result_sample_500_iter1.json")

    # Only for bbbp
    if args.dataset == "ogbg-molbbbp":
        for i in range(len(llm_generated_raw)):
            del llm_generated_raw[i]['name']

    llm_concept_value_np = np.array([[entry[key] for key in list(llm_generated_raw[0].keys())] for entry in llm_generated_raw])
    concept_value_llm = fill_nones_with_column_average(llm_concept_value_np)


    ### automatic dataloading and splitting
    if args.dataset in ["bh", "suzuki"]:
        data = GraphDataset(data_id, split_id)
        dataset = []
        for reaction in data:
            pyg_graph_list = []
            for dgl_graph in reaction[:-1]:
                pyg_graph = dgl_to_pyg(dgl_graph)
                pyg_graph_list.append(pyg_graph)

            g = reduce(merge_pyg_graphs, pyg_graph_list)
            g.y = torch.tensor(reaction[-1]).view(1,1)
            dataset.append(g)
        split_idx = 
    else:
        dataset = PygGraphPropPredDataset(name = args.dataset)
        split_idx = dataset.get_idx_split()
    
    dataset = add_attributes(dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]


    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gin-cbm':
        model = GNN_CBM(num_tasks = concept_value_llm.shape[1], gnn_type = 'gin', virtual_node = False).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []
    epoch_trace = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_trace.append(epoch)
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    # Store training curve
    kv_pair_train = zip(epoch_trace, train_curve)
    train_dict = dict(kv_pair_train)
    with open('./train_loss_bh.pkl', 'wb') as file:
        dump(train_dict, file)

    # Store test curve
    kv_pair_test = zip(epoch_trace, test_curve)
    test_dict = dict(kv_pair_test)
    with open('./test_loss_bh.pkl', 'wb') as file:
        dump(test_dict, file)

    # Store valid curve
    kv_pair_valid = zip(epoch_trace, valid_curve)
    valid_dict = dict(kv_pair_valid)
    with open('./valid_loss_bh.pkl', 'wb') as file:
        dump(valid_dict, file)


    if not args.filename == '':
        result_dict = {'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}
        with open(args.filename, 'wb') as file:
            dump(result_dict, file)

if __name__ == "__main__":
    main()
