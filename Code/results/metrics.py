import json
import os
from performance import Performance


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data
        
def save_json(json_file, json_list):
    with open(json_file, "r") as f:
        old_data = json.load(f)

    # Merge dict2 into dict1
    old_data.update(json_list)

    # Write the updated dictionary back to the same file
    with open(json_file, "w") as f:
        json.dump(old_data, f, indent=2)

def extract_human_ranks(human_anns, filter):
    human_ranks = {}
    for img, anns in human_anns.items():
        img = os.path.splitext(img)[0] + ".jpg"
        if img not in filter:
            continue
        img_data = []
        for worker_data in anns:
            objects = worker_data['objects']
            labels = []
            for object in worker_data['objects']:
                label = object['label']
                labels.append(label)
            img_data.append(labels)
        human_ranks[img] = img_data
    return human_ranks


def extract_llm_ranks(llm_anns, filter):
    llm_ranks = {}
    for img, anns in llm_anns.items():
        img = os.path.splitext(img)[0] + ".jpg"
        if img not in filter:
            continue
        llm_ranks[img] = []
        for ann in anns:
            llm_ranks[img].append(ann['label'])
    return llm_ranks

def compute_metrics(human_anns, llm_anns, dataset_name, ext=".jpg"):
    human_anns = read_json(human_anns)
    llm_anns = read_json(llm_anns)

    if ext != '.jpg':
        human_anns = {os.path.splitext(fname)[0]+".jpg": ann_values for fname,ann_values in human_anns.items()}
        llm_anns = {os.path.splitext(fname)[0]+".jpg": ann_values for fname,ann_values in llm_anns.items()}

    
    human_ranks = extract_human_ranks(human_anns, list(llm_anns.keys()))
    llm_ranks = extract_llm_ranks(llm_anns, list(human_anns.keys()))

    mean_spearman, mean_pearson, mean_mae, mean_kendall, mean_ndgc = Performance.results(human_ranks, llm_ranks)
    print(f'{dataset_name} results:')
    print(f'Mean Spearman\'s: {mean_spearman}')
    print(f'Mean Kendal-Tau\'s: {mean_kendall}')
    print(f'Mean NDGC\'s: {mean_ndgc}')
    print(f'Mean MAE\'s: {mean_mae}')
    print('------------------------------------------------------------------')



    
# Change the paths to your dataset
human_anns_explicit = "../Code/anns/human_anns/human_annotations_explicit.json"
human_anns_implicit = "../Code/anns/human_anns/human_annotations_implicit.json"

# Explicit baselines
irsr_anns = "../Code/anns/baseline_ranks/irsr_ranks.json"
assr_anns = "../Code/anns/baseline_ranks/assr_ranks.json"
sift_anns = "../Code/anns/baseline_ranks/sifr_ranks.json"
lgsr_anns = "../Code/anns/baseline_ranks/lgsr_ranks.json"

# Implicit baseline
salicon_anns = "../Code/anns/baseline_ranks/salicon_ranks.json"
osie_anns = "../Code/anns/baseline_ranks/osie_ranks.json"
mit_anns = "../Code/anns/baseline_ranks/mit_ranks.json"


compute_metrics(human_anns_implicit, salicon_anns, "SALICON")
compute_metrics(human_anns_implicit, osie_anns, "OSIE")
compute_metrics(human_anns_implicit, mit_anns, "MIT", ext=".jpeg")
compute_metrics(human_anns_explicit, irsr_anns, "IRSR", ext=".png")
compute_metrics(human_anns_explicit, assr_anns, "ASSR")
compute_metrics(human_anns_explicit, sift_anns, "SIFR")
compute_metrics(human_anns_explicit, lgsr_anns, "LGSR")




            