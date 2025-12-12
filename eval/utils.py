import json
import os
import re

def write_json(save_json_path, save_json_data):
    if os.path.exists(save_json_path):
        with open(save_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(save_json_data)
        else:
            data = [data, save_json_data]
    else:
        data = [save_json_data]

    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def extract_answer_from_response(response_text):
    m = re.search(r'\\boxed\{([^}]*)\}', response_text)
    return m.group(1) if m else ""

def build_prompt(data):
    grid_size = data.get("grid_size", [0, 0])

    # 兼容 grid_size = [rows, cols] 的情况
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        rows, cols = grid_size
        grid_str = f"{rows} × {cols}"
    else:
        # 兜底（防止脏数据）
        grid_str = str(grid_size)

    prompt = (
        f"The image shows a {grid_str} grid of objects. "
        "Most objects follow the same pattern. However, a small number of objects are anomalous "
        "and differ from the normal ones in one or more aspects, including shape, color, size, or position.\n\n"

        "Your task is to find all anomalous objects.\n\n"

        "For each anomalous object, output its coordinate in the format (row, column).  \n"
        "The top-left cell is (1, 1), rows increase downward, and columns increase to the right.\n\n"

        "Put all the final coordinates together inside a single \\boxed{ }.\n"
    )

    return prompt

