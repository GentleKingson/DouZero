import os
import argparse

from douzero.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Evaluation')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/sl/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/sl/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='')
    parser.add_argument('--ruleset', type=str, default='legacy',
            choices=['legacy', 'standard'],
            help='Ruleset: legacy (cardplay-only, default) or standard '
                 '(end-to-end bidding+cardplay with random bidding)')
    parser.add_argument('--eval_seed', type=int, default=0,
            help='Base seed for deterministic standard-mode bidding (0 = default)')
    parser.add_argument('--ruleset_config', type=str, default='',
            help='Optional YAML file with rule parameters for standard mode '
                 '(overrides RuleSet.standard() defaults). Only used with '
                 '--ruleset standard.')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    # Build a custom RuleSet from YAML if requested (shared loader with generate_eval_data).
    ruleset_obj = None
    if args.ruleset == 'standard':
        if args.ruleset_config:
            from generate_eval_data import _load_ruleset_from_config
            ruleset_obj = _load_ruleset_from_config(args.ruleset_config)
        else:
            from douzero.env.rules import RuleSet
            ruleset_obj = RuleSet.standard()

    ruleset = args.ruleset if args.ruleset != 'legacy' else None
    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers,
             ruleset=ruleset,
             eval_seed=args.eval_seed,
             ruleset_obj=ruleset_obj)
