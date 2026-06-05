"""分析 vit_query 中各组件的参数量"""
import torch
import torch.nn as nn
from model import ViTBackbone, PatchEncoder, ViTAutoregressiveExtractor7D
from neural_renderer import NeuralRasterizorStep


def count_params(module, name=""):
    """统计模块参数量"""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "name": name,
        "total": total,
        "total_m": total / 1e6,
        "trainable": trainable,
        "trainable_m": trainable / 1e6,
    }


def print_params(params, indent=0):
    """美观打印参数统计"""
    prefix = "  " * indent
    print(f"{prefix}{params['name']}:")
    print(f"{prefix}  Total: {params['total']:,} ({params['total_m']:.2f}M)")
    print(f"{prefix}  Trainable: {params['trainable']:,} ({params['trainable_m']:.2f}M)")


def analyze_vit_backbone():
    """分析 ViT Backbone"""
    print("=" * 60)
    print("ViT Backbone (ViT-B/16)")
    print("=" * 60)

    vit = ViTBackbone(img_size=224, d_model=256, pretrained=False)

    # 分模块统计
    conv_proj = count_params(vit.vit.conv_proj, "Conv Proj")
    encoder = count_params(vit.vit.encoder, "Transformer Encoder")
    feat_proj = count_params(vit.feat_proj, "Feature Projection")
    total = count_params(vit, "ViT Backbone Total")

    print_params(total)
    print()

    # 详细分析 Transformer Encoder
    print("  Transformer Encoder 详细:")
    print(f"    Layers: {len(vit.vit.encoder.layers)}")
    print(f"    Hidden dim: {vit.vit.hidden_dim}")
    print(f"    Heads: {vit.vit.encoder.layers[0].num_heads}")
    print(f"    MLP dim: {vit.vit.encoder.layers[0].mlp[0].out_features}")
    print()

    # 各层参数
    for i, layer in enumerate(vit.vit.encoder.layers[:3]):  # 只打印前3层
        layer_p = count_params(layer, f"Encoder Layer {i}")
        print_params(layer_p, indent=2)

    print(f"  ... (共 {len(vit.vit.encoder.layers)} 层)")
    print()

    # ViT 各组件
    print_params(conv_proj, indent=1)
    print_params(feat_proj, indent=1)

    return total


def analyze_patch_encoder():
    """分析 Patch Encoder (小 CNN)"""
    print("\n" + "=" * 60)
    print("Patch Encoder (CNN)")
    print("=" * 60)

    patch_enc = PatchEncoder(patch_size=64, d_model=256)
    total = count_params(patch_enc, "Patch Encoder Total")

    print_params(total)
    print()

    # 各层详细
    for i, layer in enumerate(patch_enc.conv):
        if isinstance(layer, nn.Conv2d):
            p = count_params(layer, f"Conv {i//2 + 1}")
            print_params(p, indent=1)

    proj_p = count_params(patch_enc.proj, "Projection")
    print_params(proj_p, indent=1)

    return total


def analyze_canvas_encoder():
    """分析 Canvas Encoder"""
    print("\n" + "=" * 60)
    print("Canvas Encoder (CNN)")
    print("=" * 60)

    # 直接构造相同结构
    canvas_enc = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
        nn.GELU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(256),
    )

    total = count_params(canvas_enc, "Canvas Encoder Total")
    print_params(total)
    print()

    conv_idx = 0
    for layer in canvas_enc:
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            p = count_params(layer, f"Conv {conv_idx}")
            print_params(p, indent=1)

    return total


def analyze_full_model():
    """分析完整模型"""
    print("\n" + "=" * 60)
    print("Full ViTAutoregressiveExtractor7D Model")
    print("=" * 60)

    model = ViTAutoregressiveExtractor7D(
        image_size=224, max_seq_len=48, d_model=256,
        hidden_dim=256, patch_size=64, pretrained=False
    )

    # 分模块统计
    vit_p = count_params(model.target_backbone, "ViT Backbone")
    patch_enc_p = count_params(model.patch_encoder, "Patch Encoder")
    canvas_enc_p = count_params(model.canvas_encoder, "Canvas Encoder")
    cursor_mlp_p = count_params(model.cursor_mlp, "Cursor MLP")
    window_mlp_p = count_params(model.window_mlp, "Window MLP")
    step_mlp_p = count_params(model.step_mlp, "Step MLP")
    attn_p = count_params(model.patch_target_attn, "Patch-Target Attention")
    gru_input_proj_p = count_params(model.gru_input_proj, "GRU Input Projection")
    gru_p = count_params(model.gru, "GRU Cell")
    head_p = count_params(model.head, "Stroke Head")
    total_p = count_params(model, "Full Model Total")

    print_params(total_p)
    print()
    print("  各组件:")
    print_params(vit_p, indent=1)
    print_params(patch_enc_p, indent=1)
    print_params(canvas_enc_p, indent=1)
    print_params(cursor_mlp_p, indent=1)
    print_params(window_mlp_p, indent=1)
    print_params(step_mlp_p, indent=1)
    print_params(attn_p, indent=1)
    print_params(gru_input_proj_p, indent=1)
    print_params(gru_p, indent=1)
    print_params(head_p, indent=1)

    return total_p


def analyze_renderer():
    """分析 Neural Renderer"""
    print("\n" + "=" * 60)
    print("Neural Renderer")
    print("=" * 60)

    renderer = NeuralRasterizorStep(raster_size=128)
    total = count_params(renderer, "Neural Renderer")

    print_params(total)
    print()

    # 详细分析
    raster_unit = renderer.raster_unit
    print("  RasterUnit 详细:")
    for name, module in raster_unit.named_children():
        if "fc" in name or "conv" in name or "pixel_shuffle" in name:
            p = count_params(module, name)
            print_params(p, indent=2)

    return total


def compare_with_seq_extract():
    """与 seq_extract 对比"""
    print("\n" + "=" * 60)
    print("参数量对比: vit_query vs seq_extract")
    print("=" * 60)

    print("\n  seq_extract (估算):")
    print("    CNN Encoder: ~2.87M")
    print("    HyperLSTM Decoder: ~1.08M")
    print("    Neural Renderer: ~10.55M (冻结)")
    print("    Total Trainable: ~3.95M")
    print("    Total (全部): ~14.5M")

    print("\n  vit_query (实际):")
    model = ViTAutoregressiveExtractor7D(
        image_size=224, max_seq_len=48, d_model=256,
        hidden_dim=256, patch_size=64, pretrained=False
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"    ViT Backbone: ~86.4M (主要部分)")
    print(f"    CNNs (Patch+Canvas): ~0.44M")
    print(f"    GRU + Heads: ~0.8M")
    print(f"    Total Trainable: {trainable/1e6:.2f}M")
    print(f"    Total (全部): {total/1e6:.2f}M")

    ratio = total / 3.95e6
    print(f"\n  参数量比值: {ratio:.1f}x")


if __name__ == "__main__":
    vit_total = analyze_vit_backbone()
    patch_total = analyze_patch_encoder()
    canvas_total = analyze_canvas_encoder()
    full_total = analyze_full_model()
    renderer_total = analyze_renderer()
    compare_with_seq_extract()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ViT Backbone: {vit_total['total_m']:.2f}M")
    print(f"Patch Encoder CNN: {patch_total['total_m']:.2f}M")
    print(f"Canvas Encoder CNN: {canvas_total['total_m']:.2f}M")
    print(f"Full Model (ViT version): {full_total['total_m']:.2f}M")
    print(f"Neural Renderer: {renderer_total['total_m']:.2f}M")
