from utils import *
import os
import sys

def main(dataset_path,labels_path,output_path):
    # === Paths ===
    h5_path = dataset_path

    labels_train = os.path.join(labels_path, "train_labels.json")
    labels_val = os.path.join(labels_path, "val_labels.json")
    event_to_idx_path = os.path.join(labels_path, "event_to_idx.json")

    # === Hypermparams ===
    num_epochs = 20
    batch_size = 8
    lr = 1e-4 # learning rate
    patience = 2  # early stopping

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === Loading labels ===
    with open(labels_train, "r") as f:
        labels_train = json.load(f)
    with open(labels_val, "r") as f:
        labels_val = json.load(f)
    with open(event_to_idx_path, "r") as f:
        event_to_idx = json.load(f)

    num_classes = len(set(labels_train.values()))

    # === Datasets e Loaders ===
    train_dataset = H5VideoDataset(h5_path, labels_train)
    val_dataset = H5VideoDataset(h5_path, labels_val)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # === Loading pre-treined X3D-S and reseting last 5 layers ===
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # === Loss e Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

   
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        mean_acc, acc_per_class,precision = eval(model, val_loader, device, num_classes)

        print(f"📉 train loss: {train_loss:.4f}")
        print(f"🎯 Validation mean accuracy: {mean_acc:.4f}")
        print(f"General Precision: {precision:.4f}")
        for i, acc in enumerate(acc_per_class):
            print(f"  Class {event_to_idx[i]}: {acc:.4f}")

        # === Salvar checkpoint ===
        checkpoint_path = os.path.join(output_path, f"checkpoint_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mean_acc': mean_acc,
            'acc_per_class': acc_per_class.tolist(),
            'train_loss': train_loss,
            'precision': precision,
        }, checkpoint_path)
        print(f"💾 Checkpoint saved on: {checkpoint_path}")

        # === Early Stopping ===
        if mean_acc > best_acc:
            best_acc = mean_acc
            epochs_no_improve = 0
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹️ Early stopping: acurácia não melhorou por 2 épocas consecutivas.")
                break

    print(f"\n🏁 Training step ended. Best mean accuracy: {best_acc:.4f}")

    best_checkpoint_path = os.path.join(output_path, f"checkpoint_epoch{best_epoch}.pth")

    print(f"Melhor checkpoint salvo em: {best_checkpoint_path}")


    
if __name__ == "__main__":
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(project_root_path, "DataProcessing", "processed_tensors", "tensors.h5")
    labels_path = os.path.join(project_root_path, "DataProcessing", "labels")
    output_path = os.path.join(project_root_path, "ModelTraining", "checkpoints","exp?")
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        labels_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]   
    os.makedirs(output_path, exist_ok=True)
    main(dataset_path,labels_path,output_path)