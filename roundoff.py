import json

def round_to_four(f_data, digits=4): #digits=4 are only for Mcnemar scores rounding for everything else delete it.
    #for metric in ["accuracy", "auc", "pr_auc", "precision", "recall", "f1", "specificity", "npv"]: <---- for delong
    for entry in f_data: # <---- for brier decomposition
        for metric in ["brier_score", "reliability", "resolution", "uncertainty"]: #<---- for brier decomposition
        #if metric in f_data: #<---- for delong
            if metric in entry and isinstance(entry[metric], float): #<---- for brier decomposition
                entry[metric] = round(entry[metric], 4) #<---- for brier decomposition
            #for stat in ["mean", "std", "min", "max"]: #<---- for delong
            #    if stat in f_data[metric]: #<---- for delong
            #        val = f_data[metric][stat] #<---- for delong
            #        if isinstance(val, float): #<---- for delong
            #            f_data[metric][stat] = round(val, 4) #<---- for delong
    #if isinstance(f_data, dict): <----- for mcnemar
        #return {k: round_to_four(v, digits) for k, v in f_data.items()} <----- for mcnemar
    #elif isinstance(f_data, list): <----- for mcnemar
        #return [round_to_four(elem, digits) for elem in f_data] <----- for mcnemar
    #elif isinstance(f_data, float): <----- for mcnemar
        #return round(f_data, digits) <----- for mcnemar
    #else: <----- for mcnemar
        return f_data

if __name__ == "__main__":
    with open("Exotropia_brier_decomposition_seed98_Efficient-B0.json", "r") as f:
        f_data = json.load(f)

    f_data = round_to_four(f_data)

    with open("Exotropia_brier_decomposition_seed98_Efficient-B0_check.json", "w") as f:
        json.dump(f_data, f, indent=4)