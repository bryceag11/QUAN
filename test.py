import json

# Load the annotation file
with open('data/coco2017/annotations/instances_train2017.json', 'r') as f:
    coco_data = json.load(f)

# Print out the categories
for cat in coco_data['categories']:
    print(f"ID: {cat['id']}, Name: {cat['name']}")