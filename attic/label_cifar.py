coarse_to_fine = {
    'aquatic_mammals': ('beaver', 'dolphin', 'otter', 'seal', 'whale'),
    'fish': ('aquarium fish', 'flatfish', 'ray', 'shark', 'trout'),
    'flowers': ('orchids', 'poppies', 'roses', 'sunflowers', 'tulips'),
    'food_containers': ('bottles', 'bowls', 'cans', 'cups', 'plates'),
    'fruit_and_vegetables': ('apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'),
    'household_electrical_devices': ('clock', 'computer keyboard', 'lamp', 'telephone', 'television'),
    'household_furniture': ('bed', 'chair', 'couch', 'table', 'wardrobe'),
    'insects': ('bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'),
    'large_carnivores': ('bear', 'leopard', 'lion', 'tiger', 'wolf'),
    'large_man-made_outdoor_things': ('bridge', 'castle', 'house', 'road', 'skyscraper'),
    'large_natural_outdoor_scenes': ('cloud', 'forest', 'mountain', 'plain', 'sea'),
    'large_omnivores_and_herbivores': ('camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'),
    'medium_mammals': ('fox', 'porcupine', 'possum', 'raccoon', 'skunk'),
    'non-insect_invertebrates': ('crab', 'lobster', 'snail', 'spider', 'worm'),
    'people': ('baby', 'boy', 'girl', 'man', 'woman'),
    'reptiles': ('crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'),
    'small_mammals': ('hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'),
    'trees': ('maple', 'oak', 'palm', 'pine', 'willow'),
    'vehicles_1': ('bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'),
    'vehicles_2': ('lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')
}

coarse_to_natural = {
    'aquatic_mammals': True,
    'fish': True,
    'flowers': False,
    'food_containers': False,
    'fruit_and_vegetables': True,
    'household_electrical_devices': False,
    'household_furniture': False,
    'insects': True,
    'large_carnivores': True,
    'large_man-made_outdoor_things': False,
    'large_natural_outdoor_scenes': True,
    'large_omnivores_and_herbivores': True,
    'medium_mammals': True,
    'non-insect_invertebrates': True,
    'people': True,
    'reptiles': True,
    'small_mammals': True,
    'trees': True,
    'vehicles_1': False,
    'vehicles_2': False
}


fine_labels = [
    'apple',
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm'
]

fine_to_coarse = {}
for coarse, fines in coarse_to_fine.items():
    for fine in fines:
        fine_to_coarse[fine] = coarse

assert len(fine_to_coarse) == len(fine_labels)


def split_natural():
    natural = []
    unnatrual = []
    for i, (fine, coarse) in enumerate(fine_to_coarse.items()):
        if coarse_to_natural[coarse]:
            natural.append(i)
        else:
            unnatrual.append(i)

    return natural, unnatrual


natural_labels, unnatural_labels = split_natural()
