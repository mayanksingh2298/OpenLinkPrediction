import argparse


def add_args(parser):
    # Optimization arguments
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_tokens', type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--save')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model_debug', action='store_true')
    parser.add_argument('--mode', required=True)  # train, test, resume
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--other_lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--resume_checkpoint', type=str, default='')
    parser.add_argument('--val_interval', type=float, default=1.0)
    parser.add_argument('--save_k', type=int, default=1)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--lr_warmup' , action='store_true')
    parser.add_argument('--use_scheduler', action= 'store_true')
    parser.add_argument('--use_label_embeddings', action= 'store_true')
    parser.add_argument('--negative_samples', type=int, default=10)

    # Data arguments
    parser.add_argument('--train', type=str)
    parser.add_argument('--valid', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--labels', type=str)
    # parser.add_argument('--tokens', type=str)

    # Model arguments
    parser.add_argument('--stage1', action= 'store_true')
    parser.add_argument('--stage2', action= 'store_true')
    parser.add_argument('--use_anchor', action= 'store_true')
    parser.add_argument('--shuffle', action= 'store_true')
    parser.add_argument('--add_missing_e2', action= 'store_true')
    parser.add_argument('--leave_alt_mentions', action= 'store_true')
    parser.add_argument('--stage1_model', type=str)
    parser.add_argument('--task_type', type=str) # head, tail, both
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_str', type=str)
    parser.add_argument('--model_type', type=str, help="bert|lstm|ft", default="bert", required=False)
    parser.add_argument('--xt_results', action= 'store_true')
    parser.add_argument('--ckbc', action= 'store_true')
    parser.add_argument('--add_ht_token', action= 'store_true')
    parser.add_argument('--add_ht_embeddings', action= 'store_true')
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--max_instances', type=int)
    parser.add_argument('--limit_layers', type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--retrieve_facts', type=str)
    parser.add_argument('--round_robin', action='store_true')
    parser.add_argument('--transformer', action='store_true')
    parser.add_argument('--kg_bert', action='store_true')
    parser.add_argument('--multiply_scores', action='store_true')
    parser.add_argument('--add_scores', action='store_true')
    parser.add_argument('--limit_tokens', type=int)
    return parser
