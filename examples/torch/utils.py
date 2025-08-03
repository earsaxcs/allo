from transformers import ViTForImageClassification
from allo.ops.vit import ViTImgCls
from torchvision import transforms
import torch, torchvision
import os
import PIL

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def replace_vit_with_hf_vit(vit: ViTImgCls, hf_vit: ViTForImageClassification):
    # load vit model to my customized model above
    vit.vit.embeddings.proj.weight.data = hf_vit.vit.embeddings.patch_embeddings.projection.weight.data
    vit.vit.embeddings.proj.bias.data = hf_vit.vit.embeddings.patch_embeddings.projection.bias.data
    vit.vit.embeddings.position_embeddings.data = hf_vit.vit.embeddings.position_embeddings.data
    vit.vit.embeddings.cls_token.data = hf_vit.vit.embeddings.cls_token.data
    for i in range(len(hf_vit.vit.encoder.layer)):
        vit.vit.vit_blocks[i].attention.linear_q.weight.data = hf_vit.vit.encoder.layer[i].attention.attention.query.weight.data
        vit.vit.vit_blocks[i].attention.linear_q.bias.data = hf_vit.vit.encoder.layer[i].attention.attention.query.bias.data
        vit.vit.vit_blocks[i].attention.linear_k.weight.data = hf_vit.vit.encoder.layer[i].attention.attention.key.weight.data
        vit.vit.vit_blocks[i].attention.linear_k.bias.data = hf_vit.vit.encoder.layer[i].attention.attention.key.bias.data
        vit.vit.vit_blocks[i].attention.linear_v.weight.data = hf_vit.vit.encoder.layer[i].attention.attention.value.weight.data
        vit.vit.vit_blocks[i].attention.linear_v.bias.data = hf_vit.vit.encoder.layer[i].attention.attention.value.bias.data
        vit.vit.vit_blocks[i].attention.linear_out.weight.data = hf_vit.vit.encoder.layer[i].attention.output.dense.weight.data
        vit.vit.vit_blocks[i].attention.linear_out.bias.data = hf_vit.vit.encoder.layer[i].attention.output.dense.bias.data
        vit.vit.vit_blocks[i].ffn.fc1.weight.data = hf_vit.vit.encoder.layer[i].intermediate.dense.weight.data
        vit.vit.vit_blocks[i].ffn.fc1.bias.data = hf_vit.vit.encoder.layer[i].intermediate.dense.bias.data
        vit.vit.vit_blocks[i].ffn.fc2.weight.data = hf_vit.vit.encoder.layer[i].output.dense.weight.data
        vit.vit.vit_blocks[i].ffn.fc2.bias.data = hf_vit.vit.encoder.layer[i].output.dense.bias.data
        vit.vit.vit_blocks[i].norm1.weight.data = hf_vit.vit.encoder.layer[i].layernorm_before.weight.data
        vit.vit.vit_blocks[i].norm1.bias.data = hf_vit.vit.encoder.layer[i].layernorm_before.bias.data
        vit.vit.vit_blocks[i].norm1.eps = hf_vit.vit.encoder.layer[i].layernorm_before.eps
        vit.vit.vit_blocks[i].norm2.weight.data = hf_vit.vit.encoder.layer[i].layernorm_after.weight.data
        vit.vit.vit_blocks[i].norm2.bias.data = hf_vit.vit.encoder.layer[i].layernorm_after.bias.data
        vit.vit.vit_blocks[i].norm2.eps = hf_vit.vit.encoder.layer[i].layernorm_after.eps
    vit.vit.ln_f.weight.data = hf_vit.vit.layernorm.weight.data
    vit.vit.ln_f.bias.data = hf_vit.vit.layernorm.bias.data
    vit.vit.ln_f.eps = hf_vit.vit.layernorm.eps
    vit.classifier.dense.weight.data = hf_vit.classifier.weight.data
    vit.classifier.dense.bias.data = hf_vit.classifier.bias.data

def get_imagenet_test_data(dataset_path: str, batch_size: int = -1, input_size: int = 224, is_reverse_sample: bool = False):
    val_loader = []
    with open(os.path.join(dataset_path, "true_labels.txt"), "r") as f:
        labels = f.read().split("\n")
    tmp_file_list = [x for x in os.listdir(dataset_path) if x.split(".")[-1].lower() in ["png", "jpg", "jpeg"]]
    tmp_file_list.sort()
    if batch_size != -1:
        tmp_file_list = tmp_file_list[:batch_size] if not is_reverse_sample else tmp_file_list[-batch_size:]
    for i, file in enumerate(tmp_file_list):
        img = PIL.Image.open(os.path.join(dataset_path, file)).convert('RGB')
        size = int((256 / 224) * input_size)
        img = torchvision.transforms.Compose([
            transforms.Resize(size, interpolation=3),
            transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])(img).unsqueeze(0)
        val_loader.append((img, torch.tensor(int(labels[i])).unsqueeze(0)))
        assert(img.shape[1] == 3)
    
    return val_loader