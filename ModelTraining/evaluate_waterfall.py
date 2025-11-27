from utils import *
import os
import argparse


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="Evaluate waterfall model on HDF5 video dataset")
parser.add_argument("--h5_path", type=str,default=os.path.join(project_root,"DataProcessing","processed_tensors","tensors.h5"), help="Path to the HDF5 file containing video tensors")
parser.add_argument("--labels_path", type=str,default=os.path.join(project_root,"DataProcessing","labels"), help="Path to the JSON file containing test labels")
parser.add_argument("--models_path", type=str, required=True, help="Path to trained models directory")
parser.add_argument("--batch_size", type=int, default=8, help="Size of each evaluation batch")
parser.add_argument("--num_frames", type=int, default=13, help="Number of frames per video (13 for Kinetics-400 pre-treined X3D-S model)")
parser.add_argument("--output_path", type=str, default=os.path.join(project_root,"ExperimentsResults","Results"), help="Path to save evaluation results")
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

with open(os.path.join(args.labels_path,"test_labels.json"), "r") as f:
    test_labels = json.load(f)
    
with open(os.path.join(args.labels_path,"event_to_idx.json"), "r") as f:
    event_to_idx = json.load(f)
        
with open(os.path.join(args.labels_path,"idx_to_event.json"), "r") as f:
    idx_to_event = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {}

for model in os.listdir(args.models_path):
    model_path = os.path.join(args.models_path, model)
    model_layer = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=False)
    checkpoint = torch.load(model_path,weights_only = False, map_location=device)
    num_classes = checkpoint["model_state_dict"]["blocks.5.proj.weight"].shape[0]
    in_features = model_layer.blocks[5].proj.in_features
    model_layer.blocks[5].proj = torch.nn.Linear(in_features, num_classes)
    model_layer.load_state_dict(checkpoint["model_state_dict"])
    model_layer.to(device)
    model_index = int(model.split(".")[0]) if (model.split(".")[0]).isdigit() else "root"
    models[model_index] = model_layer.eval()
    

dataset = H5VideoDataset(args.h5_path, test_labels, num_frames=args.num_frames)
dataloader = DataLoader(dataset, shuffle=False)

labels_true, labels_pred, labels_root = predict_waterfall_model(models, dataloader, device)

sub_labels = {}

for i in range(10):
    with open(os.path.join(args.labels_path,str(i),"idx_to_event.json"), "r") as f:
        sub_labels[i] = json.load(f)
    
true_pred = []
best_pred = []
top_3_pred = []
for i in range(len(labels_true)):
    labels = np.argsort(np.array(labels_pred[i][0])) 
    sub_label = sub_labels[labels_root[i]]
    output_i = [sub_label[str(label)] for label in labels]
    output_i = [event_to_idx[label] for label in output_i]
    true_pred.append(output_i)
    best_pred.append(output_i[-1])
    top_3_pred.append(output_i[-3:])

correct_top1 = 0
correct_top3 = 0
correct_per_class = np.zeros(len(event_to_idx))
total_per_class = np.zeros(len(event_to_idx))
for i in range(len(labels_true)):
    total_per_class[labels_true[i]] += 1
    if labels_true[i] == best_pred[i]:
        correct_top1 += 1
        correct_per_class[labels_true[i]] += 1
    if labels_true[i] in top_3_pred[i]:
        correct_top3 += 1

acc = correct_top1/len(labels_true)
acc_per_class = np.divide(correct_per_class, total_per_class, out=np.zeros_like(correct_per_class), where=total_per_class != 0)
acc_mean = np.mean(np.divide(correct_per_class, total_per_class, out=np.zeros_like(correct_per_class), where=total_per_class != 0))
acc_top3 = correct_top3/len(labels_true)

results = {
    "accuracy_top1": acc,
    "accuracy_mean_per_class": acc_mean,
    "accuracy_top3": acc_top3
}

with open(os.path.join(args.output_path,"waterfall_results.json"),"w") as f:
    json.dump(results,f,indent=4)


print(f"Evaluation concluded - Mean accuracy: {acc_mean:.4f}")
print(f"Overall accuracy: {acc:.4f}, Top-3 accuracy: {acc_top3:.4f}")
print("Accuracy per class:")
for cls, acc in enumerate(acc_per_class):
    print(f"Class {idx_to_event[str(cls)]} accuracy: {acc:.4f}")