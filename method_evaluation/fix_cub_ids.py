import json

id_lookup_table = {}

for line in open("datasets/CUB_200_2011/CUB_200_2011/images.txt"):
    id = line.strip().split()[0]
    filename = line.strip().split()[1].split("/")[-1].replace(".jpg", "")
    id_lookup_table[filename] = int(id)

with open('result_cub/BESTIE_CUB_wrong_id.json', 'r') as file:
    data = json.load(file)

    for item in data:
        search_name = item["image_id"]
        if search_name not in id_lookup_table:
            raise KeyError(search_name)
        item["file_name"] = search_name
        item['image_id'] = id_lookup_table[search_name]

    json.dump(data, open('result_cub/BESTIE_CUB.json', 'w'))

