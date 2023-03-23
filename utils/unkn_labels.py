MAX_SPLIT_NUM = 4

def unknown_labels(split, dataset_config):
    '''Creates the set of unknown labels in a way that all the classes in the same splits have roughly the same number of points'''
    class_percentages = dataset_config['content']
    label_percentages = {
        i: 0.0 for i in dataset_config['learning_map_inv'].keys()}
    for label, percentage in class_percentages.items():
        mapped_label = dataset_config['learning_map'][label]
        label_percentages[mapped_label] += percentage
    del(label_percentages[-1])
    label_percentages = sorted(
        label_percentages, key=label_percentages.get, reverse=True)
    novel_classes_per_split = int(len(label_percentages)/MAX_SPLIT_NUM)
    act_splitting = [novel_classes_per_split for _ in range(MAX_SPLIT_NUM)]
    tot_num = sum(act_splitting)
    i = 0
    while tot_num != len(label_percentages):
        act_splitting[i] += 1
        i += 1
        tot_num = sum(act_splitting)
    start = sum(act_splitting[:split])
    end = start + act_splitting[split]
    return label_percentages[start:end]


def label_mapping(unknown_labels, all_labels):
    new_label = -1
    label_mapping = {}
    label_mapping_inv = {}
    for label in all_labels:
        if label not in unknown_labels:
            label_mapping[label] = new_label
            label_mapping_inv[new_label] = label
            new_label += 1
    label_mapping = {**label_mapping, **
                     {unk: new_label for unk in unknown_labels}}
    del(label_mapping[-1])
    del(label_mapping_inv[-1])
    return label_mapping, label_mapping_inv, new_label