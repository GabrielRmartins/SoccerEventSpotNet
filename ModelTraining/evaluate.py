from utils import *
import os

if __name__ == "__main__":
    import argparse

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Evaluate model on HDF5 video dataset")
    parser.add_argument("--h5_path", type=str,default=os.path.join(project_root,"Common","Tensors","tensors_strategy_1.h5"), help="Path to the HDF5 file containing video tensors")
    parser.add_argument("--labels_path", type=str,default=os.path.join(project_root,"Common","Labels","Exp_1"), help="Path to folder containing experiment labels")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--batch_size", type=int, default=8, help="Size of each evaluation batch")
    parser.add_argument("--num_frames", type=int, default=13, help="Number of frames per video (13 for Kinetics-400 pre-treined X3D-S model)")
    parser.add_argument("--output_path", type=str, default=os.path.join(project_root,"ExperimentsResults","Results","Exp_1.json"), help="Path to save evaluation results")
    args = parser.parse_args()

    # Carregar rótulos
    with open(os.path.join(args.labels_path,"test_labels.json"), "r") as f:
        test_labels_dict = json.load(f)
    
    with open(os.path.join(args.labels_path,"event_to_idx.json"), "r") as f:
        labels_dict = json.load(f)
        
    with open(os.path.join(args.labels_path,"idx_to_event.json"), "r") as f:
        idx_to_event = json.load(f)
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar dataset e dataloader
    dataset = H5VideoDataset(args.h5_path, test_labels_dict, num_frames=args.num_frames)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Carregar modelo treinado
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=False)
    num_classes = len(set(labels_dict.values()))
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = torch.nn.Linear(in_features, num_classes)

    checkpoint = torch.load(args.model_path,weights_only = False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Critério de perda
    criterion = nn.CrossEntropyLoss()

    expected_labels,predicted_labels = predict(model, dataloader, device)
    predicted_labels = np.array(predicted_labels)
    predicted_labels = np.argsort(predicted_labels, axis=1)
    top_3 = predicted_labels[:,-3:]
    best = predicted_labels[:,-1]

    acc = np.sum(best == expected_labels)/len(expected_labels)
    correct_per_class = np.zeros(num_classes)
    total_per_class = np.zeros(num_classes)
    top_3_correct = 0
    for i in range(len(expected_labels)):
        label = expected_labels[i]
        total_per_class[label] += 1
        if best[i] == label:
            correct_per_class[label] += 1
        if label in top_3[i]:
            top_3_correct += 1
    acc_per_class = np.divide(correct_per_class, total_per_class, out=np.zeros_like(correct_per_class), where=total_per_class != 0)
    mean_acc = acc_per_class.mean()
    top_3_acc = top_3_correct / len(expected_labels)
    result = {}

    result["ACC"] = acc
    result["ACC_per_class"] = {idx_to_event[str(cls)]: acc for cls, acc in enumerate(acc_per_class)}
    result["ACC_Mean"] = mean_acc
    result["ACC_Top_3"] = top_3_acc
    output_path_dir = os.path.dirname(args.output_path)
    os.makedirs(output_path_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=4)
    

    print(f"Evaluation concluded - Mean accuracy: {mean_acc:.4f}")
    print(f"Overall accuracy: {acc:.4f}, Top-3 accuracy: {top_3_acc:.4f}")
    print("Accuracy per class:")
    for cls, acc in enumerate(acc_per_class):
        print(f"Class {idx_to_event[str(cls)]} accuracy: {acc:.4f}")