def get_input_seq_template(args, seq, meta):
    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new', 'sports', 'office']:
        prompt = "query: "
        for item in seq:
            try:
                title = meta[item][0]
            except:
                import pdb; pdb.set_trace()
            prompt += "'"
            prompt += title
            prompt += "', a product from "

            category = meta[item][2]
            category = ', '.join(category.split(', ')[-2:])

            prompt += category
            prompt += ' category. \n'

    return prompt

def get_input_item_template(args, item, meta):
    try:
        title = meta[item][0]
    except:
        title = meta[item[0]][0]
    category = meta[item][2]

    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new', 'sports', 'office']:
        prompt = "query: "
        prompt += "'"
        prompt += title
        prompt += "', a product from "
        prompt += category
        prompt += ' category.'

    return prompt

def get_target_item_template(args, item, meta):
    try:
        title = meta[item][0]
    except:
        title = meta[item[0]][0]
    category = meta[item][2]

    if args.dataset_code in ['beauty', 'games', 'auto', 'toys_new', 'sports', 'office']:
        prompt = "passage: "
        prompt += "'"
        prompt += title
        prompt += "', a product from "
        prompt += category
        prompt += ' category.'

    return prompt