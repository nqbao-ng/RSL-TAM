import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import custom modules
from dataloader import IEMOCAPDataset, MELDDataset
from model import Transformer_Based_Model, MaskedNLLLoss
from vision import confuPLT

def get_test_loader(dataset_name, batch_size, windows):
    """Chỉ load tập Test để inference, không shuffle."""
    if dataset_name == 'IEMOCAP':
        testset = IEMOCAPDataset("data/iemocap_multi_features.pkl", train=False, windows=windows)
    if dataset_name == 'MELD':
        testset = MELDDataset('data/meld_multimodal_features.pkl', train=False, windows=windows)

    return DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=testset.collate_fn)

def evaluate(model, dataloader, cuda):
    """Vòng lặp thuần Forward Pass, không lưu computational graph."""
    model.eval()
    preds, labels, masks = [], [], []

    with torch.no_grad():
        for data in dataloader:
            if cuda:
                data = [d.cuda() for d in data]
            
            textf, visuf, acouf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = data
            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            
            # Forward pass
            _, _, all_prob, _, _, _, _, _ = model(
                textf, visuf, acouf, umask, qmask, lengths, 
                Self_semantic_adj, Cross_semantic_adj, Semantic_adj
            )
            
            # Xử lý dự đoán
            lp_ = all_prob.view(-1, all_prob.size()[2])
            pred_ = torch.argmax(lp_, 1)
            
            preds.append(pred_.cpu().numpy())
            labels.append(label.view(-1).cpu().numpy())
            masks.append(umask.view(-1).cpu().numpy())

    # Gộp kết quả của toàn bộ batch
    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
        
        acc = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
        f1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
        return acc, f1, labels, preds, masks
    
    return 0.0, 0.0, [], [], []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to test')
    parser.add_argument('--model_path', required=True, type=str, help='Đường dẫn tới file .pth')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--windows', type=int, default=20)
    
    # Giữ lại các tham số RL để khởi tạo mô hình không bị lỗi cấu trúc
    parser.add_argument('--rl_gamma', type=float, default=0.95)
    parser.add_argument('--rl_mu', type=float, default=0.5)
    parser.add_argument('--rl_loss_w', type=float, default=0.1)
    parser.add_argument('--rl_shift_w', type=float, default=0.5)
    
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    
    print(f"[*] Đang chạy Inference trên tập: {args.Dataset} | Device: {'GPU' if cuda else 'CPU'}")

    # 1. Cấu hình Data
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 6 if args.Dataset == 'IEMOCAP' else 7

    # 2. Khởi tạo mô hình & Load Checkpoint
    model = Transformer_Based_Model(
        args.Dataset, D_text, D_visual, D_audio, args.n_head,
        n_classes=n_classes, hidden_dim=args.hidden_dim,
        n_speakers=n_speakers, dropout=0.0, # Lúc inference dropout=0
        rl_loss_w=args.rl_loss_w, rl_gamma=args.rl_gamma,
        rl_mu=args.rl_mu, rl_shift_w=args.rl_shift_w
    )
    
    if cuda:
        model.cuda()
        
    print(f"[*] Load trọng số từ: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location='cuda' if cuda else 'cpu'))

    # 3. Chạy Inference
    test_loader = get_test_loader(args.Dataset, args.batch_size, args.windows)
    acc, f1, labels, preds, masks = evaluate(model, test_loader, cuda)

    # 4. In Báo cáo
    print("\n" + "="*40)
    print("KẾT QUẢ ĐÁNH GIÁ (TEST SET)")
    print("="*40)
    print(f"Accuracy: {acc}%")
    print(f"F1-Score: {f1}%")
    print("\n--- Chi tiết từng nhãn ---")
    print(classification_report(labels, preds, sample_weight=masks, digits=4, zero_division=0))
    
    # Hiển thị Ma trận nhầm lẫn
    confuPLT(confusion_matrix(labels, preds, sample_weight=masks).astype(int), args.Dataset)

    