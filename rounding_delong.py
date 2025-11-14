import json

def rounding_delong(input_file, output_file):

    with open(input_file, "r") as f:
        data = json.load(f)

    for key, value in data.items():
        if isinstance(value, float):
            data[key] = round(value, 4)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    rounding_delong(
        "./z_temp.json",
        "./Delong_results_MobileNetV2_seed78.json"
    )