import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import os
import sys
from termcolor import colored

# Add root to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from experiments.exp11_dynamic_routing.models import create_model
from experiments.exp11_dynamic_routing.config import get_config

def get_color(is_softmax):
    # Softmax = Red (Complex/Expensive)
    # GDN = Blue (Simple/Linear)
    return 'red' if is_softmax else 'blue'

def visualize_routing(text, model, tokenizer, device):
    model.eval()
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Forward pass with routing extraction
    with torch.no_grad():
        outputs = model(input_ids, return_routing=True)
    
    # Extract routing (batch=0)
    # Shape: [seq_len, 2] -> index 1 is softmax probability/choice
    r1 = outputs['route_layer_1'][0] 
    r2 = outputs['route_layer_2'][0]
    
    # Convert to indices (0=GDN, 1=Softmax)
    r1_idx = r1.argmax(dim=-1).cpu().numpy()
    r2_idx = r2.argmax(dim=-1).cpu().numpy()
    
    print("\n" + "="*80)
    print(f"ROUTING VISUALIZATION")
    print("="*80)
    print("Legend: " + colored("LINEAR (GDN)", "blue") + " vs " + colored("SOFTMAX (ATTN)", "red"))
    print("-" * 80)
    print(f"{'TOKEN':<20} | {'LAYER 1':<15} | {'LAYER 2':<15}")
    print("-" * 80)
    
    html_output = "<html><body style='font-family: monospace; background-color: #1a1a1a; color: #e0e0e0; padding: 20px;'>"
    html_output += "<h3>Routing Visualization</h3>"
    html_output += "<p>Legend: <span style='color: #4da6ff'>LINEAR (GDN)</span> vs <span style='color: #ff4d4d'>SOFTMAX (ATTN)</span></p>"
    html_output += "<table border='1' style='border-collapse: collapse; width: 100%; text-align: left;'><tr><th>Token</th><th>Layer 1</th><th>Layer 2</th></tr>"
    
    for i, token in enumerate(tokens):
        # Decode token for cleaner display (remove Ġ etc)
        clean_token = tokenizer.decode([input_ids[0][i]]).strip()
        if not clean_token: clean_token = token # Fallback
        
        l1_is_softmax = r1_idx[i] == 1
        l2_is_softmax = r2_idx[i] == 1
        
        l1_str = "SOFTMAX" if l1_is_softmax else "LINEAR"
        l2_str = "SOFTMAX" if l2_is_softmax else "LINEAR"
        
        l1_color = get_color(l1_is_softmax)
        l2_color = get_color(l2_is_softmax)
        
        print(f"{clean_token:<20} | "
              f"{colored(l1_str, l1_color):<24} | " # Extra padding for color codes
              f"{colored(l2_str, l2_color):<15}")
        
        # HTML
        l1_html_color = "#ff4d4d" if l1_is_softmax else "#4da6ff"
        l2_html_color = "#ff4d4d" if l2_is_softmax else "#4da6ff"
        
        html_output += f"<tr><td>{clean_token}</td><td style='color: {l1_html_color}'>{l1_str}</td><td style='color: {l2_html_color}'>{l2_str}</td></tr>"

    html_output += "</table></body></html>"
    
    return html_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_dynamic/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default="The quick brown fox jumps over the lazy dog. However, the complex nuances of quantum mechanics require deeper understanding.", help='Text to analyze')
    parser.add_argument('--output_html', type=str, default='routing_viz.html', help='Output HTML file')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please train the model first or specify correct path.")
        return

    # Load Config & Model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Force dynamic routing just in case config was saved weirdly, though it should be correct
    config.use_dynamic_routing = True 
    
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    # Visualize
    html = visualize_routing(args.text, model, tokenizer, device)
    
    with open(args.output_html, 'w') as f:
        f.write(html)
    print(f"\nSaved HTML visualization to {args.output_html}")
    print(f"Open this file in your browser to see the results.")

if __name__ == "__main__":
    main()
