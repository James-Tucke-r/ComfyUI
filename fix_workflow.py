import json
import sys

file_path = "/Users/jtucker/src/comfylocal/comfy-ltx/user/default/workflows/Sulphur 2 (GGUF).json"

with open(file_path, "r") as f:
    data = json.load(f)

# Update Node 3 (LTX2_SM_Model) to load the LoRA
for node in data["nodes"]:
    if node["id"] == 3:
        # Check if the 4th widget is 'lora'
        if node["inputs"][3]["name"] == "lora":
            node["widgets_values"][3] = "sulphur_lora_rank_768.safetensors"
            print("Updated LTX2_SM_Model to load the LoRA directly.")

# Bypass Node 27 (LoraLoaderModelOnly)
# Link 37 currently goes from Node 3 -> Node 27
# Link 36 currently goes from Node 27 -> Node 5
# We will make Link 37 go from Node 3 -> Node 5 directly.

for link in data["links"]:
    if link[0] == 37: # Link 37
        link[3] = 5 # Target Node ID is now 5 (LTX2_SM_KSampler)
        link[4] = 0 # Target Slot is 0

# Remove Link 36 entirely
data["links"] = [link for link in data["links"] if link[0] != 36]

# Disconnect Node 27
for node in data["nodes"]:
    if node["id"] == 27:
        for input_port in node["inputs"]:
            if input_port["name"] == "model":
                input_port["link"] = None
        for output_port in node["outputs"]:
            if output_port["name"] == "MODEL":
                output_port["links"] = []

# Update Node 5's input to listen to Link 37 instead of Link 36
for node in data["nodes"]:
    if node["id"] == 5:
        for input_port in node["inputs"]:
            if input_port["name"] == "model":
                input_port["link"] = 37

with open(file_path, "w") as f:
    json.dump(data, f)
print("Workflow fixed successfully.")
