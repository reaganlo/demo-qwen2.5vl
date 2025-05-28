import argparse
from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM
from gradio_helper import make_demo


def main():
    parser = argparse.ArgumentParser(description="Launch OV VLM demos")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="Qwen2.5-VL-7B-Instruct-int4-ov",
        help="Path to the model directory (default: Qwen2.5-VL-7B-Instruct-int4-ov)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Device to run the model on (default: GPU)"
    )
    
    args = parser.parse_args()
    
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    
    print(f"Loading model from: {args.model_path}")
    print(f"Using device: {args.device}")
    
    model = OVModelForVisualCausalLM.from_pretrained(args.model_path, device=args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    demo = make_demo(model, processor)

    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)


if __name__ == "__main__":
    main()