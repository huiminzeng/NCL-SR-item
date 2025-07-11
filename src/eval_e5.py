import os
import torch
from config import *
from model import *
from dataloader import *
from trainer import *

from pytorch_lightning import seed_everything


def get_text_embedding(model, text, args):
    tokens = model.tokenizer(text, max_length=256, truncation=True, padding=True, return_tensors="pt")
    tokens = {k: v.to(args.device) for k, v in tokens.items()}
    outputs = model.model(**tokens)
    embeddings = average_pool(outputs.last_hidden_state, tokens['attention_mask'])
    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings

def generate_candidates(model, test_set, meta, retrieved_data_path, args):
    # prepare test dataloader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,
                                                shuffle=False, pin_memory=True, num_workers=args.num_workers,
                                                collate_fn=collate_fn_test)

    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        print('****************** Generating Candidates for Test Set ******************')
        candidate_embeddings = calculate_all_item_embeddings(model, meta, args)
        tqdm_dataloader = tqdm(test_loader)
        for _, batch in enumerate(tqdm_dataloader):
            scores, labels = calculate_logits(model, batch, meta, candidate_embeddings, args)
            test_probs.extend(scores.tolist())
            test_labels.extend(labels)
        test_metrics = absolute_recall_mrr_ndcg_for_ks(torch.tensor(test_probs), 
                                                       torch.tensor(test_labels).view(-1), [5, 10, 20, 50])
    for k in [5, 10, 20, 50]:
        print("Recall@{}: {:.4f}, \tNDCG@{}: {:.4f}".format(k, test_metrics['Recall@'+str(k)], k, test_metrics['NDCG@'+str(k)]))
        print("="*50)

    with open(retrieved_data_path, 'wb') as f:
        pickle.dump({'test_probs': test_probs,
                    'test_labels': test_labels,
                    'test_metrics': test_metrics}, f)
        
def calculate_all_item_embeddings(model, meta, args):
    # preprare all item prompts
    candidate_prompts = []
    for item in range(1, args.num_items+1):
        candidate_text = get_target_item_template(args, item, meta)
        candidate_prompts.append(candidate_text)
    candidate_embeddings =[]
    
    with torch.no_grad():
        for i in tqdm(range(0, args.num_items, args.test_batch_size)):
            input_prompts = candidate_prompts[i: i + args.test_batch_size]
            embeddings = get_text_embedding(model, input_prompts, args)
            candidate_embeddings.append(embeddings)
        candidate_embeddings = torch.cat(candidate_embeddings)
        
    return candidate_embeddings

def calculate_logits(model, batch, meta, candidate_embeddings, args):
    seqs = batch
    batch_size = len(seqs)

    input_prompts, labels = get_batch_prompts(args, meta, seqs)
    embeddings = get_text_embedding(model, input_prompts, args)

    scores = torch.matmul(embeddings, candidate_embeddings.T)
    # 0 itme padding
    place_holder = torch.zeros((batch_size, 1)).cuda()
    scores = torch.cat([place_holder, scores], dim=-1)

    for i in range(batch_size):
        scores[i, seqs[i][:-1]] = -1e9
        scores[i, 0] = -1e9  # padding

    return scores, labels

def get_batch_prompts(args, meta, seqs):
    input_prompts = []
    labels = []
    for seq in seqs:
        input_text = get_input_seq_template(args, seq[:-1], meta)
        
        input_prompts.append(input_text)
        labels.append(seq[-1])

    return input_prompts, labels

def main(args):
    seed_everything(args.seed)
    if torch.cuda.is_available(): 
        args.device = 'cuda'
    else: 
        args.device = 'cpu'

    model = E5Model()
    model.init_non_cl()
    model.to(args.device)
    load_root = os.path.join('experiments/stage_2', 
                            args.dataset_code, args.ds_mode, 
                            'num_replace_' + str(args.num_replace), 
                            'lambda_alignment_' + str(args.mce_gamma_a),
                            'lambda_uniformity_' + str(args.mce_gamma_u),
                            'lambda_item_' + str(args.mce_gamma_s))

    print("we are loading model from: ", load_root)
    model_load_name = os.path.join(load_root, 'model.checkpoint')
    if os.path.isfile(model_load_name):
        model_checkpoint = torch.load(model_load_name)
        model.load_state_dict(model_checkpoint['state_dict'])
    else:
        exit("no model found")
    
    _, _, test_set, meta = dataloader_factory_non_cl(args, model.tokenizer)

    generate_candidates(model, test_set, meta, os.path.join(load_root, 'retrieved.pkl'), args)


if __name__ == "__main__":
    set_template(args)
    main(args)
