import torch
import clip
from torchvision import datasets, transforms
from tqdm import tqdm

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
MODEL_NAME = "ViT-L/14"
DEVICE = "cuda"
SAVE_PATH = "clip_cache/multi_prompt/cifar100_train_clip_logits.pt"

TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a low resolution photo of a {}",
    "a bright photo of a {}",
    "a picture of a {}",
    "a detailed image of a {}",
    "a sketch of a {}",
    "a natural photo of a {} in the wild",
    "a realistic image of a {}",
    "a high quality photo of a {}",
]

BATCH_SIZE = 128


def build_prompts_for_class(name):
    return [t.format(name) for t in TEMPLATES]


def main():
    print(f"Loading CLIP model: {MODEL_NAME}")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()

    test_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=preprocess)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataset = datasets.CIFAR100(root="./data", train=False, download=True)
    class_names = dataset.classes
    class_embeddings_list = []

    for class_name in class_names:
        prompts = build_prompts_for_class(class_name)
        tokens = clip.tokenize(prompts).to(DEVICE)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            class_embed = text_features.mean(dim=0)
            class_embed = class_embed / class_embed.norm()
            class_embeddings_list.append(class_embed.cpu())

    class_embeddings = torch.stack(class_embeddings_list, dim=0).to(DEVICE) 

    all_logits = []
    all_targets = []

    for images, targets in tqdm(test_loader, desc="Computing logits"):
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits_list = [image_features @ class_embeddings.T] 
            logits = torch.mean(torch.stack(logits_list), dim=0)

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    all_logits = torch.cat(all_logits, dim=0) 
    all_targets = torch.cat(all_targets, dim=0)  
    
    torch.save({"logits": all_logits, "class_names": class_names}, SAVE_PATH)

if __name__ == "__main__":
    main()
